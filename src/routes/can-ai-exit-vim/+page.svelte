<script>
	import AsideBox from "../AsideBox.svelte";
	import InlineFootnote from "../InlineFootnote.svelte";
	import MediaBox from "../MediaBox.svelte";
	import * as AsciinemaPlayer from "asciinema-player";

	import "asciinema-player/dist/bundle/asciinema-player.css";

	import { onMount } from "svelte";

	let aiExitingVimContainer;
	let hxTutoringContainer;

	onMount(async () => {
		const { create } = await import("asciinema-player");
		let players = [];
		players.push(
			create("can-ai-exit-vim/ai_exiting_vim.cast", aiExitingVimContainer, {
				speed: 0.1,
				idleTimeLimit: 0.5,
				autoPlay: true,
				loop: true,
				startAt: 3.0,
			}),
		);

		players.push(
			create("can-ai-exit-vim/hx_tutoring.cast", hxTutoringContainer, {
				speed: 1.0,
				idleTimeLimit: 0.1,
				autoPlay: true,
				loop: true,
				startAt: 2.0,
			}),
		);

		return () => {
			for (const player of players) {
				player.destroy();
			}
		};
	});
</script>

<svelte:head>
	<title>Can AI Exit Vim?</title>
</svelte:head>

<h1>Can AI Exit Vim?</h1>

March 27th 2026

<hr />

<p>
	Most AI coding agents act inside a loop where they collect user input and then
	call MCP tools that allow them to run commands and read and write files. Since
	files can be very large they are usually truncated. For example when an agent
	reads a file using Opencode the result is truncated by default to 2000 lines.
	This is great for allowing the agent to receive the maximal amount of
	information about the file contents. But sometimes I wonder if the 2,000 lines
	are overkill. After all, human programers have hacked on code with tiny
	terminals for decades without issue.
</p>

<p>
	As a sort of half serious toy project I want to see if AI models can use Vim
	to edit large files. Vim is possibly more context efficient than tools like
	opencode. Instead of dumping huge files into the context, vim constrains the
	agent to only ever see a fixed-sized 32x64 terminal window. The agent's
	context is thus transformed from a sloppy pile of file contents to a crisp
	stream of terminal window snapshots.
</p>

<p>
	My end goal is to test if the AI can use CLI text editing tools to do some
	complex edits. But first there is a rite of passage I feel obliged to perform.
	Many a young hacker has no doubt smashed their fists against their keyboard
	anguishing over that one particularly obstinant CLI command that just refuses
	to close. But can AI exit Vim?
</p>

<MediaBox>
	<img src="can-ai-exit-vim/stackoverflow.webp" />
	I'm sure all of us have been to this page at least once.
</MediaBox>

<h2>The Harness</h2>

<p>
	First I want to describe the custom agentic harness I am using to allow the
	LLMs to interact with Vim and other commands as if they were typing into a
	keyboard.
</p>

<p>
	I use the terminal emulator "pyte" to create a fixed-size 32x64 terminal
	window. I made a new "clibaby" user for the AI to use so it doesn't mess up my
	home directory or anything. The agent kicks off in bash with only two simple
	tools and a short description. For all tests I use Qwen3.5-35B-A3B-Q4_K_M,
	running on my trusty 3090 GPU using llamacpp <InlineFootnote
		>Default chat parser, temperature = 0.6</InlineFootnote
	>.
</p>

<p>
	It's very simple. When the model calls <code>read_term()</code> I render the
	text of the 32x64 terminal as a string after adding a special ansi code to the
	character where the cursor is. Anything the model inputs with
	<code>input()</code> I write directly to the terminal process after un-escaping
	the special characters. The model is running in a high strung busy loop like some
	overcaffeinated gambler. Unlike Opencode it doesn't wait for the output of commands
	to finish. It simply consumes and generates tokens as fast as it can.
</p>

<table>
	<tr>
		<th> Tool </th>
		<th> Description</th>
	</tr>
	<tr>
		<td><code>read_term()</code></td>
		<td> Returns the current terminal window content as a string</td>
	</tr>
	<tr>
		<td><code>input(keys: string)</code></td>
		<td>
			The raw input to send to the terminal. Use double backslashes to escape
			special characters. '\\x03' sends Ctrl+C, '\\x1b' sends Escape, '\\n'
			sends the newline/enter key.
		</td>
	</tr>
</table>

<h3>System Prompt:</h3>
<AsideBox>
	<small>
		<p>
			You are an agent with access to a terminal screen and terminal keyboard.
			You interact with the terminal using two tools:
		</p>

		<ol>
			<li><strong>read_term</strong> - Reads the current terminal window</li>
			<li><strong>input</strong> - Sends raw keypresses to the terminal</li>
		</ol>

		<p>
			Your goal is to complete the user's task by interacting with the terminal.
		</p>

		<p>Make sure to doubly escape special characters with backslashes.</p>

		<p><strong>Guidelines:</strong></p>
		<ul>
			<li>Always start by reading the initial terminal buffer</li>
			<li>Use input to type commands and interact with programs</li>
			<li>
				Read the terminal after sending keys to ensure the results are correct
			</li>
			<li>
				For interactive programs (vim, less, etc.), use appropriate key
				sequences
			</li>
			<li>
				Common escape sequences: \\n (Enter), \\x03 (Ctrl+C), \\x1b (Escape),
				\\b (Backspace).
			</li>
			<li>
				When you're done with the task, indicate completion by not calling any
				more tools
			</li>
		</ul>

		<p><strong>Examples:</strong></p>
		<p>
			# Run the pwd command<br />
			input('pwd\\n')<br />
			# shows your current working directory.<br />
			read_term()
		</p>

		<p>
			# Clear the terminal<br />
			input('clear\\n')
		</p>

		<p>
			# Run the whoami command<br />
			input('whoami\\n')<br />
			# shows your username<br />
			read_term()
		</p>

		<p>
			<strong>Troubleshooting:</strong><br />
			First, are you remembering to type the enter key at the end of your commands?<br
			/>
			Remember that simply typing the command into the terminal is insufficient,
			you must also type the "\\n" key for the command to be run.
		</p>
	</small>
</AsideBox>

<h2>Can it exit Vim??????</h2>

<div bind:this={aiExitingVimContainer} class="asciinema-responsive"></div>

<p>Yes.</p>

<p>
	Actually it is so fast at using and writing with Vim that I had to slow down
	the above video by TEN TIMES just to get it to a speed where it was actually
	interpretable. Very humbling. It blows my mind that this is running on a six
	year old GPU.
</p>

<p>
	Now that I know it can use Vim I won't hold back. Can it finish the <a
		href="https://docs.helix-editor.com/usage.html">helix editor</a
	> tutor file? For this test I don't print the agent's direct thinking outputs because
	it goes on for a long time.
</p>

<div bind:this={hxTutoringContainer} class="asciinema-responsive"></div>

<p>
	It fails miserably at Helix. Instead of deleting the extra characters using
	the movement keys and `x` it simply inserts the correct output and deletes the
	whole line it's meant to edit. It also does not follow the instructions listed
	in the text given by the helix tutor.
</p>

<p>
	All of these issues are a powerful reminder of the shortcomings of AI agents
	as of March 2026. Despite Qwen3.5-35B-A3B being a small model it still cannot
	adhere to simple instructions and it cannot integrate observations from a
	stream of inputs. This seems similar to the nascent issues with vision
	language action (VLA) models. VLA models are trained on millions of video
	samples and try to label and plan through real streams of pixel data. They
	can't handle the heat and fall back to simple best guesses and silly hacks
	that leave them with low rewards and lackluster performance.
</p>

<h3>Large file edits</h3>

<p>
	For a final test I set off the agent to read code from a very large project,
	Huggingface's transformers.
</p>

<p>
	<i
		>"Find the implementation of qwen3.5-35B-A3B. Do all file reads with Vim.
		Your task is to write a new file, `./minimal_impl.py` that contains a simple
		minimal implementation of qwen3.5-35B-A3B's neural net architecture using
		only torch and torch.nn as imports."</i
	>
</p>

<p>
	Qwen3.5-35B-A3B set off with a strong tempo, quickly finding the file that
	implemented it's own architecture. It dutifully scrolled through the contents
	of the file, occasionally searching for important components like "Attention"
	and jumping to them. Unfortunately after 100,000 tokens it got stuck in a loop
	and repeatedly pressed control+c and the escape key.
</p>

<p>
	I tested the same prompt in Opencode. It read all of the correct files and
	almost one-shot it except for some minor shape bugs. Opencode is impressive
	and uses the models in a close to optimal way. For now, tools like Vim aren't
	especially potent when used by LLMs. But perhaps if researchers figure out how
	to get models to be good at acting through and observing through a small
	terminal window then these skill will transfer to other domains like VLA.
</p>

<style>
	table {
		table-layout: fixed;
		margin: 10px auto;
		border-collapse: collapse;
	}

	table,
	th,
	td {
		border: 1px solid black;
	}

	th,
	td {
		padding: 0.6em;
		vertical-align: top;
	}
	.asciinema-responsive {
		width: 100%;
		max-width: 700px;
		margin: auto;
		justify-content: center;
		align-items: center;
	}
</style>
