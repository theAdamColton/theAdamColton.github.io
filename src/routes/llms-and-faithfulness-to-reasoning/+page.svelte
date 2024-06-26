<svelte:head>
	<title>LLMs and faithfulness to reasoning</title>
</svelte:head>

<div class="column">
	<div class="row">
		<img src="spock cartoon.png" alt="spock sd xl 1.0" height="400" />
	</div>

	<div id="spinningcube">
		<div style="transform: translate3d(0em, 0em, -1em)">Studious,</div>
		<div style="transform: translate3d(0em, 1em, 0em)">Diligent,</div>
		<div style="transform: translate3d(0em, 2em, 1em)">Brave,</div>
		<div style="transform: translate3d(0em, 3em, 2em)">Peaceful,</div>
		<div style="transform: translate3d(0em, 4em, 3em)">That's me:</div>
		<div style="transform: translate3d(0em, 6em, 4em)">Adam</div>
		<div style="transform: translate3d(0em, 7em, 5em)">Colton</div>
	</div>
</div>

<div class="rainbow-text-animated main-title">
	<h1>LLMs and faithfulness to reasoning</h1>
</div>

<hr />

<div class="proj-box">
	<div class="proj-box-inner">
		<div class="section-title">Evaluating the logical thinking from llms:</div>

		Humans have written a trove of step-by-step explanations and proofs. These
		exist on the internet, and some of them end up in the pre-training data for
		large language models. Thus, some modicum of step-by-stepedness exists in
		the weights of LLMs.

		<br />
		<br />

		Say we want an LLM to give an answer to a multiple choice question.

		<div class="video-container">
			<div class="video-box">
				<pre>
Human: Question: 30% of all Huhulians own
at least one TV. 24% of Huhulians who own
at least one TV own at least four TV's. What
percent of Huhulians own at least four TV's?

Choices:
(A): .084%
(B): 24%
(C): 4.67%
(D): 7.2%
(E): 15.6%
</pre>
			</div>
		</div>

		Instead of simply asking directly for the answer, we can coerce the model
		into producing a step-by-step explaination.

		<div class="video-container">
			<div class="video-box">
				<pre>
Assistant: Let's think step by step: 30% of
Huhulians own at least one TV. Of those 30%,
24% own at least four TVs. So 24% of 30%, or
0.24 x 0.3 = 0.072 = 7.2% of Huhulians own at
least four TVs. The correct answer is choice (D).

Human: Given all of the above, what's the
single, most likely answer?
Assistant: The single, most likely answer is (
D)
</pre>
			</div>
		</div>

		Prompting in this way generally leads to better scores on benchmark tasks.
		The reason why such explanations lead to better performance is addressed by
		the paper
		<a href="https://arxiv.org/abs/2307.13702"
			>Measuring Faithfulness in Chain-of-Thought Reasoning</a
		>.
		<br />

		The authors demonstrate that the information in the step-by-step thinking is
		important for the LLM to produce the correct answer. They use another LLM to
		corrupt the text in the reasoning by adding mistakes. This corruption
		changes the final answer the LLM predicts, which is good because we want to
		make sure that the model's answer depends on the reasoning. If there are
		mistakes in the reasoning we expect that the final answer to be incorrect
		because it is based off of incorrect logic.

		<br />
		<br />

		A model can produce incorrect reasoning but still come up with a correct
		answer. This is called post-hoc reasoning. Even though the answer is correct
		we don't want the model using post-hoc reasoning. Explanations should be
		useful to helping us understand why a certain conclusion is reached. If the
		model is unable to explain in simple terms why a certain answer is correct,
		how should we be expected to trust the final conclusion? As Richard Feynman
		was once quoted: "If you can't explain something in simple terms, you don't
		understand it"

		<br />
		<br />

		The authors estimate the amount of post-hoc reasoning by measuring the
		chance that the LLM changes it's answer given the corrupted prompt. For some
		benchmark tasks, they measure a significant amount of post-hoc reasoning.
		Across all benchmarks, adding mistakes to the reasoning causes a decrease in
		post-hoc reasoning. At first this made me do a double take. Why would
		mistakes decrease post-hoc reasoning? Or in other words, why would mistakes
		increase the model's faithfulness to the step-by-step reasoning?

		<br />
		<br />

		One explanation is that a mistake presents suprising information which
		causes the model to pay greater attention to the reasoning. Inversely you
		could say that the reasoning going as expected can cause the model not to
		rely on it. But how can we cause models to be more faithful to the reasoning
		when the reasoning is correct?

		<br />
		<br />

		The authors discuss some interesting ideas to address this. One way to
		ensure that the answer 100% depends on the reasoning is to get the model to
		produce a logical program which when run will output the answer. You can do
		this by getting the model to generate python code. But this would be
		difficult to use for many natural language tasks.

		<br />
		<br />

		Another method is to decompose a question into sub questions, which are then
		explained and answered in turn.

		<br />
		<br />

		Given how difficult it is to measure adherance to reasoning, the authors'
		derived metric does well at capturing how faithful an explaination is. But
		underlying all of this is the idea that there exists some overall ground
		truth reasoning that can logical explain every single problem. As Spock
		learned in the original Star Trek, cold reasoning can't be applied to
		everything.
	</div>
</div>
