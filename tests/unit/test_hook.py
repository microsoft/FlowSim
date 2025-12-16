import types
import sys
import torch
from torch.profiler import profile, ProfilerActivity

from sglang.srt.tracing.hook import register, apply_auto_profile


def test_auto_profile_multiple_kernels(monkeypatch):
    # Enable hook + debug for easier inspection
    monkeypatch.setenv("SGLANG_PROFILE_KERNELS", "1")
    monkeypatch.setenv("SGLANG_PROFILE_DEBUG", "1")

    # ---- Define one "fake kernel" to simulate a profiled operator ----
    def fake_fused_moe(
        hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        return hidden_states + router_logits.mean(dim=-1, keepdim=True)

    # ---- Attach it to a temporary module (mimics a real module path) ----
    tmp_mod = types.ModuleType("sglang_fake_kernels")
    tmp_mod.fake_fused_moe = fake_fused_moe
    sys.modules["sglang_fake_kernels"] = tmp_mod

    # ---- Register (mimic the record_function label style) ----
    register(
        "sglang_fake_kernels",
        "fake_fused_moe",
        base_name="moe.fused",
        tensor_arg_names=["hidden_states", "router_logits"],
    )

    # Apply the centralized injection
    apply_auto_profile()

    # Re-import the wrapped function
    from sglang_fake_kernels import (  # type: ignore
        fake_fused_moe as wrapped_fused_moe,
    )

    # Ensure the function is wrapped
    assert getattr(wrapped_fused_moe, "_sglang_profile_wrapped", False)

    # ---- Build inputs and run a profiler pass ----
    hidden_states = torch.randn(4, 16, dtype=torch.float32)
    router_logits = torch.randn(4, 8, dtype=torch.float32)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _ = wrapped_fused_moe(hidden_states, router_logits)

    keys = [evt.key for evt in prof.key_averages()]
    print(keys)

    # Expect something like:
    # "moe.fused|hidden_states[4x16:float32],router_logits[4x8:float32]"
    assert any(k.startswith("moe.fused|") for k in keys), keys


def test_auto_profile_dotted_attr_method():
    # ---- Define a fake module with a class method ----
    class Foo:
        def bar(self, x: torch.Tensor, y: torch.Tensor):
            return x + y

    tmp_mod = types.ModuleType("sglang_fake_methods")
    tmp_mod.Foo = Foo
    sys.modules["sglang_fake_methods"] = tmp_mod

    register(
        "sglang_fake_methods",
        "Foo.bar",
        base_name="foo.bar",
        tensor_arg_names=["x", "y"],
    )

    apply_auto_profile()

    # Ensure the method is wrapped on the class.
    from sglang_fake_methods import Foo as WrappedFoo  # type: ignore

    assert getattr(WrappedFoo.bar, "_sglang_profile_wrapped", False)

    x = torch.randn(2, 3, dtype=torch.float32)
    y = torch.randn(2, 3, dtype=torch.float32)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _ = WrappedFoo().bar(x, y)

    keys = [evt.key for evt in prof.key_averages()]
    assert any(k.startswith("foo.bar|") for k in keys), keys
