"""Native integration contract; no model, target or runtime services required."""
import unittest

from decepticon.integrations.megatron import RUNTIME_CONTRACT, runtime_middleware


class MegatronRuntimeTests(unittest.TestCase):
    def test_contract_and_native_hook_order(self):
        self.assertEqual(RUNTIME_CONTRACT, 'megatron-local-v1')
        hooks = runtime_middleware(None)
        self.assertEqual([type(hook).__name__ for hook in hooks], [
            'FixedInvestigationMiddleware', 'FixedHandoffMiddleware', 'PlanningProgressMiddleware'])
        self.assertTrue(all(type(hook).__module__.startswith('decepticon.middleware.') for hook in hooks))
