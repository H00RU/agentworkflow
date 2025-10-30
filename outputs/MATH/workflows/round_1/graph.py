class Workflow:
    def __init__(self, name, llm_config, dataset):
        from scripts.async_llm import create_llm_instance
        import workflows.template.operator as operator

        self.llm = create_llm_instance(llm_config)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        # Generate multiple solutions using programmer
        solutions = []
        for i in range(3):
            result = await self.programmer(problem=problem, analysis="None")
            if result.get("output"):
                solutions.append(result.get("output"))

        # If we have multiple solutions, use ensemble
        if len(solutions) > 1:
            ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
            return ensemble_result.get("response", solutions[0] if solutions else "No solution")
        elif solutions:
            return solutions[0]
        else:
            return "No solution found"
