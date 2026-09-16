"""Copy isolation and differentiation through standard parameter containers."""

import numpy as np
import pytest

from bilby.compat.utils import array_safe_copy


def test_array_safe_copy_numpy_aliases_and_cycles():
    array = np.array([1., 2.])
    value = {"nested": [array], "alias": array, "labels": {"a", "b"}}
    value["self"] = value
    copied = array_safe_copy(value)
    assert copied is not value
    assert copied["self"] is copied
    assert copied["nested"][0] is copied["alias"]
    copied["alias"][0] = 10
    copied["labels"].add("c")
    np.testing.assert_array_equal(array, [1., 2.])
    assert value["labels"] == {"a", "b"}


@pytest.mark.parametrize("views", [False, True])
def test_array_safe_copy_nested_tensors(views):
    torch = pytest.importorskip("torch")
    original = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    tensor = original[0] if views else original
    value = {"nested": [(tensor,)], "alias": tensor}
    value["nested"].append(value)
    copied = array_safe_copy(value)
    clone = copied["nested"][0][0]
    assert clone is copied["alias"]
    assert copied["nested"][1] is copied
    assert clone is not tensor
    assert clone.data_ptr() != tensor.data_ptr()
    assert clone.dtype == tensor.dtype and clone.device == tensor.device
    clone.add_(1)
    gradient, = torch.autograd.grad(clone.square().sum(), original)
    expected = original.new_tensor([6., 0.] if views else [6., 8.])
    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(original, original.new_tensor([2., 3.]))


def test_array_safe_copy_direct_tensor_and_tensor_set():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor(2., requires_grad=True)
    clone = array_safe_copy(tensor)
    assert clone is not tensor
    gradient, = torch.autograd.grad(clone * 3, tensor)
    torch.testing.assert_close(gradient, tensor.new_tensor(3.))
    result = array_safe_copy((set([tensor]), frozenset([tensor])))
    left, = result[0]
    right, = result[1]
    assert left is right and left is not tensor


@pytest.mark.parametrize("compiled", [False, True])
def test_array_safe_copy_jax_tracing(compiled):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def objective(original):
        array = original * 2
        value = {"nested": [(array,)], "alias": array, "labels": ["original"]}
        value["self"] = value
        copied = array_safe_copy(value)
        assert copied["self"] is copied
        assert copied["nested"][0][0] is copied["alias"]
        assert copied["alias"].dtype == array.dtype
        copied["labels"].append("changed")
        copied["alias"] = copied["alias"].at[0].add(1)
        assert value["labels"] == ["original"]
        return jnp.sum(copied["alias"] ** 2) + jnp.sum(value["alias"])

    evaluate = jax.value_and_grad(objective)
    if compiled:
        evaluate = jax.jit(evaluate)
    original = jnp.array([2., 3.])
    result, gradient = evaluate(original)
    np.testing.assert_allclose(result, 71.)
    np.testing.assert_allclose(gradient, [22., 26.])
    np.testing.assert_array_equal(original, [2., 3.])


def test_array_safe_copy_direct_jax_array():
    jnp = pytest.importorskip("jax.numpy")
    original = jnp.array([2., 3.])
    copied = array_safe_copy(original)
    assert copied is not original
    assert copied.dtype == original.dtype
    assert copied.device == original.device
    np.testing.assert_array_equal(copied, original)
