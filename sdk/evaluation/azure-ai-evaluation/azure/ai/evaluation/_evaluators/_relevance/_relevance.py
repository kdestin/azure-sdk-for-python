# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import logging
import math
import os
from typing import Dict, Union, List

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from ..._common.utils import reformat_conversation_history, reformat_agent_response

from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase

logger = logging.getLogger(__name__)


class RelevanceEvaluator(PromptyEvaluatorBase):
    """
    Evaluates relevance score for a given query and response or a multi-turn conversation, including reasoning.

    The relevance measure assesses the ability of answers to capture the key points of the context.
    High relevance scores signify the AI system's understanding of the input and its capability to produce coherent
    and contextually appropriate outputs. Conversely, low relevance scores indicate that generated responses might
    be off-topic, lacking in context, or insufficient in addressing the user's intended queries. Use the relevance
    metric when evaluating the AI system's performance in understanding the input and generating contextually
    appropriate responses.

    Relevance scores range from 1 to 5, with 1 being the worst and 5 being the best.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the relevance evaluator. Default is 3.
    :type threshold: int

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START relevance_evaluator]
            :end-before: [END relevance_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a RelevanceEvaluator with a query, response, and context.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START relevance_evaluator]
            :end-before: [END relevance_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call RelevanceEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_relevance_evaluator]
            :end-before: [END threshold_relevance_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a RelevanceEvaluator with a query, response, and context.

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    # Constants must be defined within eval's directory to be save/loadable
    _PROMPTY_FILE = "relevance.prompty"
    _RESULT_KEY = "relevance"

    id = "azureai://built-in/evaluators/relevance"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=3):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold,
            _higher_is_better=self._higher_is_better,
        )

    @overload
    def __call__(
        self,
        *,
        query: str,
        response: str,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate groundedness for given input of query, response, context

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :return: The relevance score.
        :rtype: Dict[str, float]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate relevance for a conversation

        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The relevance score.
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Evaluate relevance. Accepts either a query and response for a single evaluation,
        or a conversation for a multi-turn evaluation. If the conversation has more than one turn,
        the evaluator will aggregate the results of each turn.

        :keyword query: The query to be evaluated. Mutually exclusive with the `conversation` parameter.
        :paramtype query: Optional[str]
        :keyword response: The response to be evaluated. Mutually exclusive with the `conversation` parameter.
        :paramtype response: Optional[str]
        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The relevance score.
        :rtype: Union[Dict[str, Union[str, float]], Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]]
        """
        return super().__call__(*args, **kwargs)

    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a relevance evaluation.

        :param eval_input: The input to the evaluator. Expected to contain
        whatever inputs are needed for the _flow method, including context
        and other fields depending on the child class.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Only text conversation inputs are supported.",
                internal_message="Only text conversation inputs are supported.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.CONVERSATION,
            )
        if not isinstance(eval_input["query"], str):
            eval_input["query"] = reformat_conversation_history(eval_input["query"], logger)
        if not isinstance(eval_input["response"], str):
            eval_input["response"] = reformat_agent_response(eval_input["response"], logger)
        llm_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        score = math.nan

        if isinstance(llm_output, dict):
            score = float(llm_output.get("score", math.nan))
            reason = llm_output.get("explanation", "")
            # Parse out score and reason from evaluators known to possess them.
            binary_result = self._get_binary_result(score)
            return {
                self._result_key: float(score),
                f"gpt_{self._result_key}": float(score),
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_result": binary_result,
                f"{self._result_key}_threshold": self._threshold,
            }

        binary_result = self._get_binary_result(score)
        return {
            self._result_key: float(score),
            f"gpt_{self._result_key}": float(score),
            f"{self._result_key}_result": binary_result,
            f"{self._result_key}_threshold": self._threshold,
        }
