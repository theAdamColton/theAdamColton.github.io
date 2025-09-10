<script context="module">
	let id = 0;
</script>

<script>
	id = id + 1;
	let checkbox_id = `checkbox-${id}`;

	export let title = "note";

	let checked;
	let parent;
	let popupRect;
	let windowWidth;

	let popupWidthPx = 500;
	$: {
		if (windowWidth) {
			popupWidthPx = Math.min(popupWidthPx, windowWidth - 50);
		}
	}

	let popupLeftAdjustment = 0;
	$: {
		if (!checked || !parent || !windowWidth || !popupRect) {
			popupLeftAdjustment = 0;
		} else {
			let popupLeft = parent.offsetLeft + parent.offsetWidth;
			let popupWidth = popupRect.width;
			let distanceFromButtonToLeft = windowWidth - popupLeft;
			let adjustmentLeft =
				distanceFromButtonToLeft > popupWidth
					? 0
					: distanceFromButtonToLeft - popupWidth;
			popupLeftAdjustment = adjustmentLeft;
		}
	}
</script>

<svelte:window bind:innerWidth={windowWidth} />
<span class="ref" bind:this={parent}>
	<input type="checkbox" id={checkbox_id} bind:checked /><label
		for={checkbox_id}><span class="reftitle">[{title}]</span></label
	>
	<span
		class="refbody"
		style={`left:${popupLeftAdjustment}px; width:${popupWidthPx}px`}
		bind:contentRect={popupRect}
	>
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
		position: absolute;
		left: -10px;
		bottom: 20px;
		border: 2px double orange;
		padding: 2px;
		background-color: #d4fbff;
		z-index: 9999;
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
