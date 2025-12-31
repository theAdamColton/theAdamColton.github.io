<script>
	import hljs from 'highlight.js/lib/core';
	import python from 'highlight.js/lib/languages/python';
	import 'highlight.js/styles/atom-one-dark.css';

	export let code = '';
	export let language = 'python';

	let highlightedCode = '';

	hljs.registerLanguage('python', python);

	// Reactive statement to update highlighting when code or language changes
	$: {
		if (code && language) {
			try {
				const result = hljs.highlight(code, { language, ignoreIllegals: true });
				highlightedCode = result.value;
			} catch (e) {
				console.error(e);
				// Fallback to unhighlighted code on error
				highlightedCode = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
			}
		}
	}
</script>

<pre><code class="hljs language-{language}">{@html highlightedCode}</code></pre>

<style>
	pre {
		border-radius: 5px;
		overflow-x: auto;
	}
</style>