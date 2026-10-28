"""Native fork-owned runtime for Megatron's isolated job containers.

Opt in with DECEPTICON_MEGATRON_RUNTIME=1. Trusted launch scope/workflow
selection and all authorization/evidence gates remain required.
"""
RUNTIME_CONTRACT = 'megatron-local-v1'


def runtime_middleware(backend):
    from decepticon.middleware.fixed_investigation import FixedInvestigationMiddleware
    from decepticon.middleware.fixed_handoff import FixedHandoffMiddleware
    from decepticon.middleware.planning_progress import PlanningProgressMiddleware

    return [FixedInvestigationMiddleware(backend), FixedHandoffMiddleware(backend), PlanningProgressMiddleware()]
