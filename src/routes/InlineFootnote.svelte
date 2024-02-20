<script context="module">
	let id = 0;
</script>

<script>
	id = id + 1;
	let checkbox_id = `checkbox-${id}`;

	export let title = "note";
</script>

<span class="ref">
	<input type="checkbox" id={checkbox_id} /><label for={checkbox_id}
		><span class="reftitle">[{title}]</span></label
	>
	<span class="refbody">
		<slot />
	</span>
</span>

<style>
	.reftitle {
		position: relative;
		left: -1px;
		bottom: 0.4em;
		color: #365693;
		font-size: 0.8em;
		font-weight: 700;
		text-decoration: underline;
		cursor: pointer;
		padding-right: 3px;
	}
	.ref {
		position: relative;
		vertical-align: baseline;
	}
	.refbody {
		display: block;
		min-width: 15em;
		max-width: 20em;
		position: absolute;
		left: max(-15em, -8vw);
		bottom: 20px;
		border: 2px double orange;
		padding: 2px;
		background-color: #d4fbff;
		word-wrap: break-word;
		z-index: 9999;
		overflow: auto;
	}
	input[type="checkbox"] ~ span {
		display: none;
	}
	input[type="checkbox"]:checked ~ span {
		display: block;
	}
	input[type="checkbox"] {
		display: none;
	}
</style>
