def input_node(state):
    print("--- Input Node ---")
    user_query = state.get("user_query")
    conversation_history = state.get("conversation_history", [])
    tenant_data = state.get("tenant_data", {})
    awaiting_tenant_input_field = state.get("awaiting_tenant_input_field")
    current_field_index = state.get("current_field_index", 0)

    updated_state = state.copy()
    if "conversation_history" not in updated_state or updated_state["conversation_history"] is None:
        updated_state["conversation_history"] = []

    if awaiting_tenant_input_field:
        tenant_data[awaiting_tenant_input_field] = user_query
        updated_state["tenant_data"] = tenant_data
        current_field_index += 1
        updated_state["current_field_index"] = current_field_index
        updated_state["next_node"] = "tenant_action_node"
    else:
        print("Input Node - Normal user query (not awaiting specific input)")
        updated_state["user_query"] = user_query
        updated_state["conversation_history"] = conversation_history
        updated_state["next_node"] = "intent_router_node"

    print("Input Node State (Updated):", updated_state)
    return updated_state