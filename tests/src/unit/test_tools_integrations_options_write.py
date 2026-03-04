"""Unit tests for integration options write tooling."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.exceptions import ToolError


def _capture_tools(mock_client: Any) -> dict[str, Any]:
    """Register integration tools and return the captured functions."""
    from ha_mcp.tools.tools_integrations import register_integration_tools

    registered_tools: dict[str, Any] = {}

    def capture_tool(**kwargs):
        def decorator(fn):
            registered_tools[fn.__name__] = fn
            return fn

        return decorator

    mock_mcp = MagicMock()
    mock_mcp.tool = capture_tool
    register_integration_tools(mock_mcp, mock_client)
    return registered_tools


class TestIntegrationOptionsWrite:
    """Tests for ha_set_integration_options."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.get_config_entry = AsyncMock(
            return_value={
                "entry_id": "entry-1",
                "domain": "versatile_thermostat",
                "title": "VT TV-stue",
                "state": "loaded",
            }
        )
        client.start_options_flow = AsyncMock()
        client.submit_options_flow_step = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_set_integration_options_success_type_step(self, mock_client):
        """Successful VT type-step update should return diff and verification."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.side_effect = [
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["type", "finalize"],
            },
            {
                "type": "menu",
                "flow_id": "flow-2",
                "step_id": "menu",
                "menu_options": ["type", "finalize"],
            },
        ]
        mock_client.submit_options_flow_step.side_effect = [
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "type",
                "data_schema": [
                    {
                        "name": "underlying_entity_ids",
                        "description": {"suggested_value": ["climate.tado_x_tv_stue"]},
                    }
                ],
            },
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["type", "finalize"],
            },
            {"type": "create_entry", "flow_id": "flow-1", "step_id": "finalize"},
            {
                "type": "form",
                "flow_id": "flow-2",
                "step_id": "type",
                "data_schema": [
                    {
                        "name": "underlying_entity_ids",
                        "description": {
                            "suggested_value": [
                                "climate.0x449fdafffe7c4205",
                                "climate.0x983268fffe059ee1",
                            ]
                        },
                    }
                ],
            },
        ]

        result = await registered_tools["ha_set_integration_options"](
            entry_id="entry-1",
            step="type",
            options_patch={
                "underlying_entity_ids": [
                    "climate.0x449fdafffe7c4205",
                    "climate.0x983268fffe059ee1",
                ]
            },
        )

        assert result["success"] is True
        assert result["applied"] is True
        assert result["verified"] is True
        assert result["verification_method"] == "flow_suggested"
        assert result["diff"] == [
            {
                "key": "underlying_entity_ids",
                "before": ["climate.tado_x_tv_stue"],
                "after": [
                    "climate.0x449fdafffe7c4205",
                    "climate.0x983268fffe059ee1",
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_set_integration_options_noop(self, mock_client):
        """Applying the same values should return applied=false."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.return_value = {
            "type": "menu",
            "flow_id": "flow-1",
            "step_id": "menu",
            "menu_options": ["type", "finalize"],
        }
        mock_client.submit_options_flow_step.return_value = {
            "type": "form",
            "flow_id": "flow-1",
            "step_id": "type",
            "data_schema": [
                {
                    "name": "underlying_entity_ids",
                    "description": {"suggested_value": ["climate.tado_x_tv_stue"]},
                }
            ],
        }

        result = await registered_tools["ha_set_integration_options"](
            entry_id="entry-1",
            step="type",
            options_patch={"underlying_entity_ids": ["climate.tado_x_tv_stue"]},
        )

        assert result["success"] is True
        assert result["applied"] is False
        assert result["diff"] == []

    @pytest.mark.asyncio
    async def test_set_integration_options_rejects_unsupported_step(self, mock_client):
        """Unsupported integration-step combinations should fail clearly."""
        registered_tools = _capture_tools(mock_client)

        with pytest.raises(ToolError) as exc_info:
            await registered_tools["ha_set_integration_options"](
                entry_id="entry-1",
                step="window",
                options_patch={"use_window_central_config": True},
            )

        error_data = json.loads(str(exc_info.value))
        assert error_data["success"] is False
        assert error_data["error"]["code"] == "VALIDATION_INVALID_PARAMETER"
        assert "Unsupported integration options step" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_set_integration_options_rejects_invalid_key(self, mock_client):
        """Unsupported keys should be rejected before submission."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.return_value = {
            "type": "menu",
            "flow_id": "flow-1",
            "step_id": "menu",
            "menu_options": ["type", "finalize"],
        }
        mock_client.submit_options_flow_step.return_value = {
            "type": "form",
            "flow_id": "flow-1",
            "step_id": "type",
            "data_schema": [
                {
                    "name": "underlying_entity_ids",
                    "description": {"suggested_value": ["climate.tado_x_tv_stue"]},
                }
            ],
        }

        with pytest.raises(ToolError) as exc_info:
            await registered_tools["ha_set_integration_options"](
                entry_id="entry-1",
                step="type",
                options_patch={"temperature_sensor_entity_id": "sensor.foo"},
            )

        error_data = json.loads(str(exc_info.value))
        assert error_data["success"] is False
        assert error_data["error"]["code"] == "VALIDATION_INVALID_PARAMETER"
        assert "Unsupported option key" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_set_integration_options_rejects_step_unavailable_for_entry(
        self, mock_client
    ):
        """Entry-specific menu availability should be enforced before submission."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.return_value = {
            "type": "menu",
            "flow_id": "flow-1",
            "step_id": "menu",
            "menu_options": ["main", "features", "presence"],
        }

        with pytest.raises(ToolError) as exc_info:
            await registered_tools["ha_set_integration_options"](
                entry_id="entry-1",
                step="type",
                options_patch={"underlying_entity_ids": ["climate.foo"]},
            )

        error_data = json.loads(str(exc_info.value))
        assert error_data["success"] is False
        assert error_data["error"]["code"] == "CONFIG_VALIDATION_FAILED"
        assert "not available for config entry" in error_data["error"]["message"]
        assert error_data["available_steps"] == ["main", "features", "presence"]

    @pytest.mark.asyncio
    async def test_set_integration_options_surfaces_ha_form_errors(self, mock_client):
        """HA form validation errors should become MCP validation errors."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.return_value = {
            "type": "menu",
            "flow_id": "flow-1",
            "step_id": "menu",
            "menu_options": ["type", "finalize"],
        }
        mock_client.submit_options_flow_step.side_effect = [
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "type",
                "data_schema": [
                    {
                        "name": "underlying_entity_ids",
                        "description": {"suggested_value": ["climate.tado_x_tv_stue"]},
                    }
                ],
            },
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "type",
                "errors": {"base": "invalid_underlying_entity"},
                "data_schema": [
                    {
                        "name": "underlying_entity_ids",
                        "description": {"suggested_value": ["climate.tado_x_tv_stue"]},
                    }
                ],
            },
        ]

        with pytest.raises(ToolError) as exc_info:
            await registered_tools["ha_set_integration_options"](
                entry_id="entry-1",
                step="type",
                options_patch={"underlying_entity_ids": ["climate.invalid"]},
            )

        error_data = json.loads(str(exc_info.value))
        assert error_data["success"] is False
        assert error_data["error"]["code"] == "CONFIG_VALIDATION_FAILED"
        assert "Home Assistant rejected the options update" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_set_integration_options_accepts_schema_only_field_names(
        self, mock_client
    ):
        """Fields without suggested/default values should still be writable."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.side_effect = [
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["main", "finalize"],
            },
            {
                "type": "menu",
                "flow_id": "flow-2",
                "step_id": "menu",
                "menu_options": ["main", "finalize"],
            },
        ]
        mock_client.submit_options_flow_step.side_effect = [
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "main",
                "data_schema": [
                    {
                        "name": "last_seen_temperature_sensor_entity_id",
                    },
                    {
                        "name": "cycle_min",
                        "description": {"suggested_value": 5},
                    },
                ],
            },
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["main", "finalize"],
            },
            {"type": "create_entry", "flow_id": "flow-1", "step_id": "finalize"},
            {
                "type": "form",
                "flow_id": "flow-2",
                "step_id": "main",
                "data_schema": [
                    {
                        "name": "last_seen_temperature_sensor_entity_id",
                        "description": {"suggested_value": "sensor.trv_last_seen"},
                    },
                    {
                        "name": "cycle_min",
                        "description": {"suggested_value": 5},
                    },
                ],
            },
        ]

        result = await registered_tools["ha_set_integration_options"](
            entry_id="entry-1",
            step="main",
            options_patch={"last_seen_temperature_sensor_entity_id": "sensor.trv_last_seen"},
        )

        assert result["success"] is True
        assert result["applied"] is True
        assert result["verified"] is True
        assert result["diff"] == [
            {
                "key": "last_seen_temperature_sensor_entity_id",
                "before": None,
                "after": "sensor.trv_last_seen",
            }
        ]

    @pytest.mark.asyncio
    async def test_set_integration_options_accepts_features_step_keys(self, mock_client):
        """New VT features-step adapters should accept supported keys."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.side_effect = [
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["features", "finalize"],
            },
            {
                "type": "menu",
                "flow_id": "flow-2",
                "step_id": "menu",
                "menu_options": ["features", "finalize"],
            },
        ]
        mock_client.submit_options_flow_step.side_effect = [
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "features",
                "data_schema": [
                    {
                        "name": "use_auto_start_stop_feature",
                        "description": {"suggested_value": False},
                    },
                    {
                        "name": "use_window_feature",
                        "description": {"suggested_value": False},
                    },
                ],
            },
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["features", "finalize"],
            },
            {"type": "create_entry", "flow_id": "flow-1", "step_id": "finalize"},
            {
                "type": "form",
                "flow_id": "flow-2",
                "step_id": "features",
                "data_schema": [
                    {
                        "name": "use_auto_start_stop_feature",
                        "description": {"suggested_value": True},
                    },
                    {
                        "name": "use_window_feature",
                        "description": {"suggested_value": False},
                    },
                ],
            },
        ]

        result = await registered_tools["ha_set_integration_options"](
            entry_id="entry-1",
            step="features",
            options_patch={"use_auto_start_stop_feature": True},
        )

        assert result["success"] is True
        assert result["applied"] is True
        assert result["verified"] is True
        assert result["diff"] == [
            {
                "key": "use_auto_start_stop_feature",
                "before": False,
                "after": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_set_integration_options_accepts_tpi_step_keys(self, mock_client):
        """VT tpi-step adapters should accept supported keys."""
        registered_tools = _capture_tools(mock_client)

        mock_client.start_options_flow.side_effect = [
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["tpi", "finalize"],
            },
            {
                "type": "menu",
                "flow_id": "flow-2",
                "step_id": "menu",
                "menu_options": ["tpi", "finalize"],
            },
        ]
        mock_client.submit_options_flow_step.side_effect = [
            {
                "type": "form",
                "flow_id": "flow-1",
                "step_id": "tpi",
                "data_schema": [
                    {
                        "name": "tpi_coef_int",
                        "description": {"suggested_value": 0.6},
                    },
                    {
                        "name": "minimal_activation_delay",
                        "description": {"suggested_value": 10},
                    },
                ],
            },
            {
                "type": "menu",
                "flow_id": "flow-1",
                "step_id": "menu",
                "menu_options": ["tpi", "finalize"],
            },
            {"type": "create_entry", "flow_id": "flow-1", "step_id": "finalize"},
            {
                "type": "form",
                "flow_id": "flow-2",
                "step_id": "tpi",
                "data_schema": [
                    {
                        "name": "tpi_coef_int",
                        "description": {"suggested_value": 0.8},
                    },
                    {
                        "name": "minimal_activation_delay",
                        "description": {"suggested_value": 10},
                    },
                ],
            },
        ]

        result = await registered_tools["ha_set_integration_options"](
            entry_id="entry-1",
            step="tpi",
            options_patch={"tpi_coef_int": 0.8},
        )

        assert result["success"] is True
        assert result["applied"] is True
        assert result["verified"] is True
        assert result["diff"] == [
            {
                "key": "tpi_coef_int",
                "before": 0.6,
                "after": 0.8,
            }
        ]
