# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import os
import time
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

    def _create_default_initial_workflow(self, graph_path: str):
        """Create default initial workflow templates when not found"""
        import json

        # Create template directory
        template_dir = os.path.join(graph_path, "template")
        os.makedirs(template_dir, exist_ok=True)

        # Create __init__.py for template
        template_init = os.path.join(template_dir, "__init__.py")
        with open(template_init, 'w') as f:
            f.write("")

        # Create symbolic link to AFlow operators
        operator_py_path = os.path.join(template_dir, "operator.py")
        aflow_operator_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "scripts", "operators.py"
        ))
        if not os.path.exists(operator_py_path):
            try:
                os.symlink(aflow_operator_path, operator_py_path)
                logger.info(f"Created symlink: {operator_py_path} -> {aflow_operator_path}")
            except OSError:
                # If symlink fails, copy the file
                import shutil
                shutil.copy(aflow_operator_path, operator_py_path)
                logger.info(f"Copied operator.py to {operator_py_path}")

        operator_json_path = os.path.join(template_dir, "operator.json")
        if not os.path.exists(operator_json_path):
            operator_descriptions = {
                "Custom": {
                    "description": "A flexible operator that takes custom instructions and processes input accordingly",
                    "interface": "Custom(input: str, instruction: str) -> Dict[str, Any]"
                },
                "ScEnsemble": {
                    "description": "Self-Consistency Ensemble operator that aggregates multiple solutions",
                    "interface": "ScEnsemble(solutions: List[str], problem: str) -> Dict[str, Any]"
                },
                "Programmer": {
                    "description": "Generates and executes Python code to solve problems",
                    "interface": "Programmer(problem: str, analysis: str = 'None') -> Dict[str, Any]"
                },
                "AnswerGenerate": {
                    "description": "Generates direct answers to questions",
                    "interface": "AnswerGenerate(input: str) -> Tuple[str, str]"
                },
                "CustomCodeGenerate": {
                    "description": "Generates code with custom instructions",
                    "interface": "CustomCodeGenerate(problem: str, entry_point: str, instruction: str) -> Dict[str, Any]"
                },
                "Test": {
                    "description": "Tests generated code solutions",
                    "interface": "Test(solution: str, entry_point: str) -> Union[List[Dict], str]"
                }
            }
            with open(operator_json_path, 'w', encoding='utf-8') as f:
                json.dump(operator_descriptions, f, indent=2, ensure_ascii=False)
            logger.info(f"Created operator.json at {operator_json_path}")

        # Create round_1 directory
        round_1_dir = os.path.join(graph_path, "round_1")
        os.makedirs(round_1_dir, exist_ok=True)

        # Create __init__.py
        init_file = os.path.join(round_1_dir, "__init__.py")
        with open(init_file, 'w') as f:
            f.write("")

        # Create default prompt.py based on question type
        prompt_file = os.path.join(round_1_dir, "prompt.py")
        if self.type == "math":
            prompt_content = '''# Math problem solving prompts
SOLVE_PROMPT = """Solve the following math problem step by step. Show your work clearly.

Problem: """

ANALYZE_PROMPT = """Analyze the following math problem and identify the key concepts and steps needed.

Problem: """
'''
        elif self.type == "code":
            prompt_content = '''# Code generation prompts
CODE_PROMPT = """Write Python code to solve the following problem. Include a main function.

Problem: """

TEST_PROMPT = """Review the following code and suggest improvements.

Code: """
'''
        else:  # qa
            prompt_content = '''# Question answering prompts
ANSWER_PROMPT = """Answer the following question based on the given context.

Question: """

REASONING_PROMPT = """Provide reasoning for your answer to the following question.

Question: """
'''

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        logger.info(f"Created prompt.py at {prompt_file}")

        # Create default graph.py based on question type and operators
        graph_file = os.path.join(round_1_dir, "graph.py")

        if self.type == "math":
            if "Programmer" in self.operators:
                graph_content = '''class Workflow:
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
'''
            else:
                graph_content = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        from scripts.async_llm import create_llm_instance
        import workflows.template.operator as operator
        import workflows.round_1.prompt as prompt_custom

        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Generate multiple solutions
        solutions = []
        for i in range(3):
            result = await self.custom(input=problem, instruction="Solve this math problem step by step:\\n")
            if result.get("response"):
                solutions.append(result.get("response"))

        # Use ensemble to select best solution
        if len(solutions) > 1:
            ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
            return ensemble_result.get("response", solutions[0])
        elif solutions:
            return solutions[0]
        else:
            return "No solution found"
'''
        elif self.type == "code":
            graph_content = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        from scripts.async_llm import create_llm_instance
        import workflows.template.operator as operator

        self.llm = create_llm_instance(llm_config)
        self.code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str = "main"):
        # Generate code solution
        solutions = []
        for i in range(2):
            result = await self.code_generate(
                problem=problem,
                entry_point=entry_point,
                instruction="Write clean, efficient code:\\n"
            )
            if result.get("response"):
                solutions.append(result.get("response"))

        if len(solutions) > 1:
            ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
            return ensemble_result.get("response", solutions[0])
        elif solutions:
            return solutions[0]
        else:
            return "# No solution generated"
'''
        else:  # qa
            graph_content = '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        from scripts.async_llm import create_llm_instance
        import workflows.template.operator as operator

        self.llm = create_llm_instance(llm_config)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        # Generate multiple answers
        solutions = []
        for i in range(3):
            result = await self.answer_generate(input=problem)
            if isinstance(result, tuple):
                solutions.append(result[0])
            elif result.get("response"):
                solutions.append(result.get("response"))

        # Use ensemble for final answer
        if len(solutions) > 1:
            ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)
            return ensemble_result.get("response", solutions[0])
        elif solutions:
            return solutions[0]
        else:
            return "No answer found"
'''

        # No need to replace dataset name anymore since we use relative imports

        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write(graph_content)
        logger.info(f"Created graph.py at {graph_file}")

        logger.info(f"Successfully created default initial workflow for {self.dataset} ({self.type})")

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)

            # Check if round_1 workflow exists, if not create default one
            round_1_graph = os.path.join(graph_path, "round_1", "graph.py")
            if not os.path.exists(round_1_graph):
                logger.info(f"Initial workflow not found, creating default workflow for {self.dataset}")
                self._create_default_initial_workflow(graph_path)

            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

        # Create a loop until the generated graph meets the check conditions
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])

            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
            )

            # Replace ActionNode with AsyncLLM and XmlFormatter
            try:
                # Create XmlFormatter based on GraphOptimize model
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                
                # Call the LLM with formatter
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt, 
                    graph_formatter
                )
                
                # If we reach here, response is properly formatted and validated
                logger.info(f"Graph optimization response received successfully")
            except FormatError as e:
                # Handle format validation errors
                logger.error(f"Format error in graph optimization: {str(e)}")
                # Try again with a fallback approach - direct call with post-processing
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                
                # Try to extract fields using basic parsing
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error("Failed to extract fields from raw response, retrying...")
                    continue

            # Check if the modification meets the conditions
            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            # If `check` is True, break the loop; otherwise, regenerate the graph
            if check:
                break

        # Save the graph and evaluate
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False)

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re
            
            # Initialize result dictionary with default values
            result = {
                "modification": "",
                "graph": "",
                "prompt": ""
            }
            
            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()
            
            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None
            
            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self):
        rounds = [1]  # You can choose the rounds you want to test here.
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)