<script>
	// We use a random ID so you can have multiple footnotes on one page
	const id = Math.random().toString(36).slice(2, 9);
	const anchorName = `--pop-${id}`;
</script>

<span class="footnote-wrapper">
	<button
		popovertarget="pop-{id}"
		style:anchor-name={anchorName}
		class="link-button"
		aria-label="View footnote"
	>
		[note]
	</button>

	<div
		popover
		id="pop-{id}"
		style:position-anchor={anchorName}
		class="footnote-content"
	>
		<slot />
	</div>
</span>

<style>
	.footnote-wrapper {
		display: inline-block;
	}

	.link-button {
		background: none;
		border: none;
		padding: 0;
		color: #0066cc;
		text-decoration: underline;
		cursor: pointer;
		font: small;
		vertical-align: super;
	}

	.footnote-content {
		position: absolute;

		inset: auto;
		margin: 0;

		position-area: top;

		margin-bottom: 10px;

		border: 1px solid #ccc;
		border-radius: 6px;
		padding: 10px;
		background: white;
		box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
		width: max-content;
		max-width: 250px;
		font-size: 0.9em;
		color: #333;
	}
</style>
