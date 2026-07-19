from core.runtime.state_propagation import propagate_execution_state


def test_propagate_execution_state_preserves_existing_object_identity():
    execution_state = object()
    previous_state = {"execution_state": execution_state, "steps": 1}
    node_update = {"plan": "keep going"}

    result = propagate_execution_state(previous_state, node_update)

    assert result is not node_update
    assert "execution_state" not in node_update
    assert result["execution_state"] is execution_state
    assert result["plan"] == "keep going"
    assert node_update == {"plan": "keep going"}
    assert previous_state == {"execution_state": execution_state, "steps": 1}


def test_propagate_execution_state_respects_explicit_replacement():
    original_execution_state = object()
    replacement_execution_state = object()
    previous_state = {"execution_state": original_execution_state}
    node_update = {"execution_state": replacement_execution_state, "steps": 2}

    result = propagate_execution_state(previous_state, node_update)

    assert result is node_update
    assert result["execution_state"] is replacement_execution_state


def test_propagate_execution_state_is_noop_without_prior_execution_state():
    node_update = {"plan": "noop"}

    result = propagate_execution_state({"steps": 1}, node_update)

    assert result is node_update


def test_propagate_execution_state_passes_through_non_mapping_updates():
    node_update = "__end__"

    result = propagate_execution_state({"execution_state": object()}, node_update)

    assert result == "__end__"


def test_propagate_execution_state_does_not_recreate_object():
    execution_state = object()
    previous_state = {"execution_state": execution_state}

    result = propagate_execution_state(previous_state, {})

    assert result["execution_state"] is execution_state