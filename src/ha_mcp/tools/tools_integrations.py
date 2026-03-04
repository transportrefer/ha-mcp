"""
Integration management tools for Home Assistant MCP server.

This module provides tools to list, enable, disable, and delete Home Assistant
integrations (config entries) via the REST and WebSocket APIs.
"""

import logging
from copy import deepcopy
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..errors import ErrorCode, create_error_response
from .helpers import exception_to_structured_error, log_tool_usage, raise_tool_error
from .util_helpers import coerce_bool_param

logger = logging.getLogger(__name__)


INTEGRATION_OPTION_ADAPTERS: dict[str, dict[str, dict[str, Any]]] = {
    "versatile_thermostat": {
        "main": {
            "allowed_keys": {
                "external_temperature_sensor_entity_id",
                "temp_min",
                "temp_max",
                "step_temperature",
                "name",
                "temperature_sensor_entity_id",
                "last_seen_temperature_sensor_entity_id",
                "cycle_min",
                "device_power",
                "use_main_central_config",
                "use_central_mode",
                "used_by_controls_central_boiler",
            },
            "verification_method": "flow_suggested",
        },
        "features": {
            "allowed_keys": {
                "use_window_feature",
                "use_motion_feature",
                "use_power_feature",
                "use_presence_feature",
                "use_central_boiler_feature",
                "use_heating_failure_detection_feature",
                "use_auto_start_stop_feature",
            },
            "verification_method": "flow_suggested",
        },
        "presets": {
            "allowed_keys": {
                "use_presets_central_config",
            },
            "verification_method": "flow_suggested",
        },
        "presence": {
            "allowed_keys": {
                "presence_sensor_entity_id",
                "use_presence_central_config",
            },
            "verification_method": "flow_suggested",
        },
        "advanced": {
            "allowed_keys": {
                "safety_delay_min",
                "safety_min_on_percent",
                "safety_default_on_percent",
                "use_advanced_central_config",
            },
            "verification_method": "flow_suggested",
        },
        "lock": {
            "allowed_keys": {
                "lock_code",
                "lock_users",
                "lock_automations",
                "use_lock_central_config",
            },
            "verification_method": "flow_suggested",
        },
        "type": {
            "allowed_keys": {
                "underlying_entity_ids",
                "ac_mode",
                "sync_device_internal_temp",
                "auto_regulation_mode",
                "auto_regulation_dtemp",
                "auto_regulation_periode_min",
                "auto_fan_mode",
                "auto_regulation_use_device_temp",
            },
            "verification_method": "flow_suggested",
        },
    }
}


def _format_options_flow_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the stable public subset of an options-flow response."""
    formatted = {
        "type": result.get("type"),
        "step_id": result.get("step_id"),
        "flow_id": result.get("flow_id"),
    }
    if "data_schema" in result:
        formatted["data_schema"] = result.get("data_schema", [])
    if "menu_options" in result:
        formatted["menu_options"] = result.get("menu_options", [])
    if "errors" in result:
        formatted["errors"] = result.get("errors", {})
    if "description_placeholders" in result:
        formatted["description_placeholders"] = result.get(
            "description_placeholders", {}
        )
    return formatted


def _extract_schema_values(flow_result: dict[str, Any]) -> dict[str, Any]:
    """Extract the current/suggested/default values from a flow schema."""
    values: dict[str, Any] = {}
    for field in flow_result.get("data_schema", []):
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not isinstance(name, str) or not name:
            continue
        if "suggested_value" in field:
            values[name] = field.get("suggested_value")
        elif isinstance(field.get("description"), dict) and "suggested_value" in field["description"]:
            values[name] = field["description"].get("suggested_value")
        elif "default" in field:
            values[name] = field.get("default")
    return values


def _extract_schema_field_names(flow_result: dict[str, Any]) -> set[str]:
    """Extract all field names from a flow schema."""
    names: set[str] = set()
    for field in flow_result.get("data_schema", []):
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _build_diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a stable per-key diff between two dictionaries."""
    diff: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value != after_value:
            diff.append({"key": key, "before": before_value, "after": after_value})
    return diff


def _normalize_options_patch(
    options_patch: dict[str, Any],
) -> dict[str, Any]:
    """Return a validated shallow copy of options_patch."""
    if not isinstance(options_patch, dict) or not options_patch:
        raise_tool_error(
            create_error_response(
                ErrorCode.VALIDATION_INVALID_PARAMETER,
                "options_patch must be a non-empty object.",
                context={"parameter": "options_patch"},
            )
        )
    return deepcopy(options_patch)


def _get_adapter(domain: str, step: str) -> dict[str, Any]:
    """Return the adapter config for a supported integration/step pair."""
    domain_adapters = INTEGRATION_OPTION_ADAPTERS.get(domain, {})
    adapter = domain_adapters.get(step)
    if adapter is None:
        raise_tool_error(
            create_error_response(
                ErrorCode.VALIDATION_INVALID_PARAMETER,
                f"Unsupported integration options step: {domain}.{step}",
                context={"domain": domain, "step": step},
                suggestions=[
                    "Use ha_get_integration_options(..., include_options_flow=True) to inspect the available flow",
                    "Limit writes to currently supported integration/step combinations",
                ],
            )
        )
    return adapter


async def _open_options_step(client: Any, entry_id: str, step: str) -> dict[str, Any]:
    """Start an options flow and navigate to the requested step."""
    flow = await client.start_options_flow(entry_id)
    if flow.get("type") == "menu":
        flow = await client.submit_options_flow_step(
            flow["flow_id"], {"next_step_id": step}
        )

    if flow.get("step_id") != step:
        raise_tool_error(
            create_error_response(
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"Options flow did not open the requested step '{step}'.",
                context={
                    "entry_id": entry_id,
                    "requested_step": step,
                    "actual_step": flow.get("step_id"),
                    "flow_type": flow.get("type"),
                },
            )
        )
    return flow


def _validate_patch_keys(
    patch: dict[str, Any],
    adapter: dict[str, Any],
    schema_field_names: set[str],
    *,
    domain: str,
    step: str,
    strict_keys: bool,
) -> None:
    """Validate patch keys against adapter policy and current flow schema."""
    adapter_keys = set(adapter.get("allowed_keys", set()))
    unknown_keys = sorted(key for key in patch if key not in adapter_keys)
    unavailable_keys = sorted(key for key in patch if key not in schema_field_names)

    if unknown_keys and strict_keys:
        raise_tool_error(
            create_error_response(
                ErrorCode.VALIDATION_INVALID_PARAMETER,
                f"Unsupported option key(s) for {domain}.{step}: {', '.join(unknown_keys)}",
                context={
                    "domain": domain,
                    "step": step,
                    "unsupported_keys": unknown_keys,
                },
            )
        )
    if unavailable_keys:
        raise_tool_error(
            create_error_response(
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"Requested option key(s) are not available on the current {domain}.{step} flow step: {', '.join(unavailable_keys)}",
                context={
                    "domain": domain,
                    "step": step,
                    "available_keys": sorted(schema_field_names),
                    "requested_keys": sorted(patch),
                },
            )
        )


async def _verify_step_values(
    client: Any,
    entry_id: str,
    step: str,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Re-open the step and compare readback values to expected ones."""
    verification_flow = await _open_options_step(client, entry_id, step)
    readback = _extract_schema_values(verification_flow)
    verified = all(readback.get(key) == value for key, value in expected.items())
    return readback, verified


async def _finalize_options_flow(
    client: Any, flow_result: dict[str, Any]
) -> dict[str, Any]:
    """Finalize an options flow when HA returns to the menu after a step submit."""
    if flow_result.get("type") == "menu" and "finalize" in flow_result.get(
        "menu_options", []
    ):
        flow_result = await client.submit_options_flow_step(
            flow_result["flow_id"], {"next_step_id": "finalize"}
        )
    return flow_result


def register_integration_tools(mcp: Any, client: Any, **kwargs: Any) -> None:
    """Register integration management tools with the MCP server."""

    @mcp.tool(annotations={"idempotentHint": True, "readOnlyHint": True, "tags": ["integration"], "title": "Get Integration"})
    @log_tool_usage
    async def ha_get_integration(
        entry_id: Annotated[
            str | None,
            Field(
                description="Config entry ID to get details for. "
                "If omitted, lists all integrations.",
                default=None,
            ),
        ] = None,
        query: Annotated[
            str | None,
            Field(
                description="When listing, fuzzy search by domain or title.",
                default=None,
            ),
        ] = None,
        domain: Annotated[
            str | None,
            Field(
                description="Filter by integration domain (e.g. 'template', 'group'). "
                "When set, includes the full options/configuration for each entry.",
                default=None,
            ),
        ] = None,
        include_options: Annotated[
            bool | str,
            Field(
                description="Include the options object for each entry. "
                "Automatically enabled when domain filter is set. "
                "Useful for auditing template definitions and helper configurations.",
                default=False,
            ),
        ] = False,
    ) -> dict[str, Any]:
        """
        Get integration (config entry) information - list all or get a specific one.

        Without an entry_id: Lists all configured integrations with optional filters.
        With an entry_id: Returns detailed information including full options/configuration.

        Use this to audit existing configurations (e.g. template sensor Jinja code).
        When creating new functionality, prefer UI-based helpers over templates when possible.

        EXAMPLES:
        - List all integrations: ha_get_integration()
        - Search integrations: ha_get_integration(query="zigbee")
        - Get specific entry: ha_get_integration(entry_id="abc123")
        - List template entries with definitions: ha_get_integration(domain="template")
        - List all with options: ha_get_integration(include_options=True)

        STATES: 'loaded' (running), 'setup_error', 'setup_retry', 'not_loaded',
        'failed_unload', 'migration_error'.

        RETURNS (when listing):
        - entries: List of integrations with domain, title, state, capabilities
        - state_summary: Count of entries in each state
        - When domain filter or include_options is set, each entry includes the 'options' object

        RETURNS (when getting specific entry):
        - entry: Full config entry details including options/configuration
        """
        try:
            include_opts = coerce_bool_param(include_options, "include_options", default=False)
            # Auto-enable options when domain filter is set
            if domain is not None:
                include_opts = True

            # If entry_id provided, get specific config entry
            if entry_id is not None:
                try:
                    result = await client.get_config_entry(entry_id)
                    return {"success": True, "entry_id": entry_id, "entry": result}
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "not found" in error_msg.lower():
                        raise_tool_error(create_error_response(
                            ErrorCode.RESOURCE_NOT_FOUND,
                            f"Config entry not found: {entry_id}",
                            context={"entry_id": entry_id},
                            suggestions=[
                                "Use ha_get_integration() without entry_id to see all config entries",
                            ],
                        ))
                    raise

            # List mode - get all config entries
            # Use REST API endpoint for config entries
            response = await client._request(
                "GET", "/config/config_entries/entry"
            )

            if not isinstance(response, list):
                raise_tool_error(create_error_response(
                    ErrorCode.SERVICE_CALL_FAILED,
                    "Unexpected response format from Home Assistant",
                    context={"response_type": type(response).__name__},
                ))

            entries = response

            # Apply domain filter before formatting
            if domain:
                domain_lower = domain.strip().lower()
                entries = [e for e in entries if e.get("domain", "").lower() == domain_lower]

            # Format entries for response
            formatted_entries = []
            for entry in entries:
                formatted_entry = {
                    "entry_id": entry.get("entry_id"),
                    "domain": entry.get("domain"),
                    "title": entry.get("title"),
                    "state": entry.get("state"),
                    "source": entry.get("source"),
                    "supports_options": entry.get("supports_options", False),
                    "supports_unload": entry.get("supports_unload", False),
                    "disabled_by": entry.get("disabled_by"),
                }

                # Include options when requested (for auditing template definitions, etc.)
                if include_opts:
                    formatted_entry["options"] = entry.get("options", {})

                # Include pref_disable_new_entities and pref_disable_polling if present
                if "pref_disable_new_entities" in entry:
                    formatted_entry["pref_disable_new_entities"] = entry[
                        "pref_disable_new_entities"
                    ]
                if "pref_disable_polling" in entry:
                    formatted_entry["pref_disable_polling"] = entry[
                        "pref_disable_polling"
                    ]

                formatted_entries.append(formatted_entry)

            # Apply fuzzy search filter if query provided
            if query and query.strip():
                from ..utils.fuzzy_search import calculate_ratio

                # Perform fuzzy search with both exact and fuzzy matching
                matches = []
                query_lower = query.strip().lower()

                for entry in formatted_entries:
                    domain_lower = entry['domain'].lower()
                    title_lower = entry['title'].lower()

                    # Check for exact substring matches first (highest priority)
                    if query_lower in domain_lower or query_lower in title_lower:
                        # Exact substring match gets score of 100
                        matches.append((100, entry))
                    else:
                        # Try fuzzy matching on domain and title separately
                        domain_score = calculate_ratio(query_lower, domain_lower)
                        title_score = calculate_ratio(query_lower, title_lower)
                        best_score = max(domain_score, title_score)

                        if best_score >= 70:  # threshold for fuzzy matches
                            matches.append((best_score, entry))

                # Sort by score descending
                matches.sort(key=lambda x: x[0], reverse=True)
                formatted_entries = [match[1] for match in matches]

            # Group by state for summary
            state_summary: dict[str, int] = {}
            for entry in formatted_entries:
                state = entry.get("state", "unknown")
                state_summary[state] = state_summary.get(state, 0) + 1

            result_data: dict[str, Any] = {
                "success": True,
                "total": len(formatted_entries),
                "entries": formatted_entries,
                "state_summary": state_summary,
                "query": query if query else None,
            }
            if domain:
                result_data["domain_filter"] = domain.strip().lower()
            return result_data

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to get integrations: {e}")
            exception_to_structured_error(
                e,
                suggestions=[
                    "Verify Home Assistant connection is working",
                    "Check that the API is accessible",
                    "Ensure your token has sufficient permissions",
                ],
            )

    @mcp.tool(
        annotations={
            "idempotentHint": True,
            "readOnlyHint": True,
            "tags": ["integration"],
            "title": "Get Integration Options",
        }
    )
    @log_tool_usage
    async def ha_get_integration_options(
        entry_id: Annotated[
            str,
            Field(
                description="Config entry ID for which to retrieve persisted options and options-flow metadata."
            ),
        ],
        include_options_flow: Annotated[
            bool | str,
            Field(
                description="When true, also start the integration's options flow and return the initial flow metadata.",
                default=True,
            ),
        ] = True,
    ) -> dict[str, Any]:
        """
        Get persisted integration options and the initial options-flow shape.

        This is a generic inspection tool for integrations that expose settings
        through Home Assistant's Options Flow.
        """
        try:
            include_flow = coerce_bool_param(
                include_options_flow, "include_options_flow", default=True
            )

            entry = await client.get_config_entry(entry_id)
            result: dict[str, Any] = {
                "success": True,
                "entry_id": entry_id,
                "domain": entry.get("domain"),
                "title": entry.get("title"),
                "state": entry.get("state"),
                "supports_options": entry.get("supports_options", False),
                "options": entry.get("options", {}),
            }

            if not include_flow:
                return result

            flow = await client.start_options_flow(entry_id)
            result["options_flow"] = _format_options_flow_result(flow)
            return result

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to get integration options: {e}")
            exception_to_structured_error(
                e,
                context={"entry_id": entry_id},
                suggestions=[
                    "Use ha_get_integration() to confirm the config entry exists",
                    "Check whether this integration exposes an options flow in Home Assistant",
                ],
            )

    @mcp.tool(
        annotations={
            "destructiveHint": True,
            "idempotentHint": True,
            "tags": ["integration"],
            "title": "Set Integration Options",
        }
    )
    @log_tool_usage
    async def ha_set_integration_options(
        entry_id: Annotated[str, Field(description="Config entry ID to update.")],
        step: Annotated[
            str,
            Field(
                description="Options-flow step to update, for example 'presence' or 'type'."
            ),
        ],
        options_patch: Annotated[
            dict[str, Any],
            Field(
                description="Partial object containing the fields to update on the selected step."
            ),
        ],
        strict_keys: Annotated[
            bool | str,
            Field(
                description="When true, reject keys not explicitly supported for the integration/step adapter.",
                default=True,
            ),
        ] = True,
        verify: Annotated[
            bool | str,
            Field(
                description="When true, re-open the flow step and compare suggested values after applying the patch.",
                default=True,
            ),
        ] = True,
    ) -> dict[str, Any]:
        """
        Update integration settings through Home Assistant's config-entry options flow.

        This is a generic write tool with integration-step adapters for safe payload
        shaping and verification. Use ha_get_integration_options() first to inspect
        the flow before applying changes.
        """
        try:
            strict_keys_bool = coerce_bool_param(
                strict_keys, "strict_keys", default=True
            )
            verify_bool = coerce_bool_param(verify, "verify", default=True)
            patch = _normalize_options_patch(options_patch)

            entry = await client.get_config_entry(entry_id)
            domain = entry.get("domain")
            title = entry.get("title")
            if not isinstance(domain, str) or not domain:
                raise_tool_error(
                    create_error_response(
                        ErrorCode.CONFIG_NOT_FOUND,
                        f"Config entry {entry_id} does not expose a valid integration domain.",
                        context={"entry_id": entry_id},
                    )
                )

            adapter = _get_adapter(domain, step)
            step_flow = await _open_options_step(client, entry_id, step)
            before = _extract_schema_values(step_flow)
            schema_field_names = _extract_schema_field_names(step_flow)
            _validate_patch_keys(
                patch,
                adapter,
                schema_field_names,
                domain=domain,
                step=step,
                strict_keys=strict_keys_bool,
            )

            payload = deepcopy(before)
            payload.update(patch)
            diff = _build_diff(before, payload)
            if not diff:
                return {
                    "success": True,
                    "entry_id": entry_id,
                    "domain": domain,
                    "title": title,
                    "step": step,
                    "applied": False,
                    "before": before,
                    "after": before,
                    "diff": [],
                    "verified": True,
                    "verification_method": "none",
                    "warnings": [],
                }

            submit_result = await client.submit_options_flow_step(
                step_flow["flow_id"], payload
            )
            if submit_result.get("type") == "form":
                raise_tool_error(
                    create_error_response(
                        ErrorCode.CONFIG_VALIDATION_FAILED,
                        f"Home Assistant rejected the options update for {domain}.{step}.",
                        context={
                            "entry_id": entry_id,
                            "domain": domain,
                            "step": step,
                            "errors": submit_result.get("errors", {}),
                        },
                        suggestions=[
                            "Review the flow errors in the response context",
                            "Call ha_get_integration_options(..., include_options_flow=True) to inspect the current step schema",
                        ],
                    )
                )
            submit_result = await _finalize_options_flow(client, submit_result)

            warnings: list[str] = []
            after = deepcopy(payload)
            verified = False
            verification_method = "none"

            if verify_bool:
                verification_method = adapter.get("verification_method", "none")
                if verification_method == "flow_suggested":
                    after, verified = await _verify_step_values(
                        client, entry_id, step, patch
                    )
                    if not verified:
                        warnings.append(
                            "Options flow update completed, but the follow-up readback did not fully match the requested values."
                        )
                else:
                    warnings.append(
                        "No verification strategy is implemented for this integration-step adapter."
                    )
            else:
                warnings.append("Verification skipped at caller request.")

            return {
                "success": True,
                "entry_id": entry_id,
                "domain": domain,
                "title": title,
                "step": step,
                "applied": True,
                "before": before,
                "after": after,
                "diff": diff,
                "verified": verified,
                "verification_method": verification_method,
                "warnings": warnings,
            }

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to set integration options: {e}")
            exception_to_structured_error(
                e,
                context={"entry_id": entry_id, "step": step},
                suggestions=[
                    "Use ha_get_integration_options(..., include_options_flow=True) to inspect the target step",
                    "Confirm the integration/step pair is currently supported by ha_set_integration_options",
                ],
            )

    @mcp.tool(
        annotations={
            "destructiveHint": True,
            "tags": ["integration"],
            "title": "Set Integration Enabled",
        }
    )
    @log_tool_usage
    async def ha_set_integration_enabled(
        entry_id: Annotated[str, Field(description="Config entry ID")],
        enabled: Annotated[
            bool | str, Field(description="True to enable, False to disable")
        ],
    ) -> dict[str, Any]:
        """Enable/disable integration (config entry).

        Use ha_get_integration() to find entry IDs.
        """
        try:
            enabled_bool = coerce_bool_param(enabled, "enabled")

            message = {
                "type": "config_entries/disable",
                "entry_id": entry_id,
                "disabled_by": None if enabled_bool else "user",
            }

            result = await client.send_websocket_message(message)

            if not result.get("success"):
                error_msg = result.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise_tool_error(create_error_response(
                    ErrorCode.SERVICE_CALL_FAILED,
                    f"Failed to {'enable' if enabled_bool else 'disable'} integration: {error_msg}",
                    context={"entry_id": entry_id},
                ))

            # Get updated entry info
            require_restart = result.get("result", {}).get("require_restart", False)

            if require_restart:
                note = "Home Assistant restart required for changes to take effect."
            else:
                note = "Integration has been loaded." if enabled_bool else "Integration has been unloaded."

            return {
                "success": True,
                "message": f"Integration {'enabled' if enabled_bool else 'disabled'} successfully",
                "entry_id": entry_id,
                "require_restart": require_restart,
                "note": note,
            }

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to set integration enabled: {e}")
            exception_to_structured_error(e, context={"entry_id": entry_id})

    @mcp.tool(
        annotations={
            "destructiveHint": True,
            "tags": ["integration"],
            "title": "Delete Config Entry",
        }
    )
    @log_tool_usage
    async def ha_delete_config_entry(
        entry_id: Annotated[str, Field(description="Config entry ID")],
        confirm: Annotated[
            bool | str, Field(description="Must be True to confirm deletion")
        ] = False,
    ) -> dict[str, Any]:
        """Delete config entry permanently. Requires confirm=True.

        Use ha_get_integration() to find entry IDs.
        """
        try:
            confirm_bool = coerce_bool_param(confirm, "confirm", default=False)

            if not confirm_bool:
                raise_tool_error(create_error_response(
                    ErrorCode.VALIDATION_INVALID_PARAMETER,
                    "Deletion not confirmed. Set confirm=True to proceed.",
                    context={
                        "entry_id": entry_id,
                        "warning": "This will permanently delete the config entry. This cannot be undone.",
                    },
                ))

            message = {
                "type": "config_entries/delete",
                "entry_id": entry_id,
            }

            result = await client.send_websocket_message(message)

            if not result.get("success"):
                error_msg = result.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise_tool_error(create_error_response(
                    ErrorCode.SERVICE_CALL_FAILED,
                    f"Failed to delete config entry: {error_msg}",
                    context={"entry_id": entry_id},
                ))

            # Get result info
            require_restart = result.get("result", {}).get("require_restart", False)

            return {
                "success": True,
                "message": "Config entry deleted successfully",
                "entry_id": entry_id,
                "require_restart": require_restart,
                "note": (
                    "The integration has been permanently removed."
                    if not require_restart
                    else "Home Assistant restart required to complete removal."
                ),
            }

        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete config entry: {e}")
            exception_to_structured_error(e, context={"entry_id": entry_id})
