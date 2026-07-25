import json

import httpx
import pytest

from ai.care import (
    AnthropicCareProvider,
    CareAdvisor,
    CareRecommendation,
    OpenAICareProvider,
    PlantIdentity,
    ProviderNotConfigured,
)
from core.crypto import encrypt_value
from models.setting import Setting


class FakeProvider:
    default_model = "test-model"

    def __init__(self):
        self.calls = []

    async def recommend(self, plant, **kwargs):
        self.calls.append((plant, kwargs))
        return CareRecommendation(tasks=[], care_summary="Healthy")


def save_setting(db, user_id, key, value):
    db.add(Setting(user_id=user_id, key=key, value=encrypt_value(value)))
    db.commit()


@pytest.mark.asyncio
async def test_advisor_uses_explicit_provider(db, test_user):
    anthropic = FakeProvider()
    openai = FakeProvider()
    save_setting(db, test_user.id, "ai_provider", "openai")
    save_setting(db, test_user.id, "openai_api_key", "openai-secret")

    result = await CareAdvisor(
        db, {"anthropic": anthropic, "openai": openai}
    ).recommend(PlantIdentity(common_name="Fern"), user_id=test_user.id)

    assert result.care_summary == "Healthy"
    assert not anthropic.calls
    assert openai.calls[0][1]["api_key"] == "openai-secret"


@pytest.mark.asyncio
async def test_advisor_preserves_legacy_anthropic_setting(db, test_user):
    anthropic = FakeProvider()
    save_setting(db, test_user.id, "anthropic_api_key", "legacy-secret")

    await CareAdvisor(db, {"anthropic": anthropic}).recommend(
        PlantIdentity(common_name="Fern"), user_id=test_user.id
    )

    assert anthropic.calls[0][1]["api_key"] == "legacy-secret"


@pytest.mark.asyncio
async def test_advisor_does_not_infer_openai_provider(db, test_user):
    save_setting(db, test_user.id, "openai_api_key", "openai-secret")

    with pytest.raises(ProviderNotConfigured):
        await CareAdvisor(db, {"openai": FakeProvider()}).recommend(
            PlantIdentity(common_name="Fern"), user_id=test_user.id
        )


@pytest.mark.asyncio
async def test_openai_provider_uses_responses_schema():
    request_body = {}

    async def handle(request):
        request_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "care_summary": "Keep evenly moist.",
                                        "tasks": [
                                            {
                                                "task_type": "water",
                                                "frequency_days": 7,
                                                "notes": None,
                                            }
                                        ],
                                    }
                                ),
                            }
                        ],
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await OpenAICareProvider(client).recommend(
            PlantIdentity(common_name="Fern"),
            api_key="secret",
            model="gpt-5.6-luna",
            safety_identifier="safe-user",
        )

    assert result.tasks[0].frequency_days == 7
    assert request_body["store"] is False
    assert request_body["reasoning"] == {"effort": "none"}
    assert request_body["text"]["format"]["strict"] is True
    assert request_body["safety_identifier"] == "safe-user"


@pytest.mark.asyncio
async def test_anthropic_provider_preserves_messages_contract():
    request_body = {}

    async def handle(request):
        request_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "text": json.dumps(
                            {"care_summary": "Bright light.", "tasks": []}
                        )
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await AnthropicCareProvider(client).recommend(
            PlantIdentity(common_name="Cactus"),
            api_key="secret",
            model="claude-test",
            safety_identifier="unused",
        )

    assert result.care_summary == "Bright light."
    assert request_body["model"] == "claude-test"
    assert request_body["messages"][0]["role"] == "user"


def test_ai_care_requires_configuration(client, auth_headers):
    response = client.post(
        "/plants/species/ai-care",
        json={"common_name": "Fern"},
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Choose an AI provider")
