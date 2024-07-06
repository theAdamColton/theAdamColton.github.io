<script context="module">
	let id = 0;
</script>

<script>
	id = id + 1;
	let checkbox_id = `checkbox-${id}`;

	export let title = "note";

	let parent;
	let popupRect;
	let windowWidth;

	let popupLeftAdjustment = 0;
	$: {
		if (!parent || !windowWidth || !popupRect) {
			popupLeftAdjustment = 0;
		} else {
			let popupLeft = parent.offsetLeft + parent.offsetWidth;
			let distanceFromLeft = windowWidth - popupLeft;
			let popupWidth = popupRect.width;
			let adjustmentLeft =
				distanceFromLeft > popupWidth ? 0 : distanceFromLeft - popupWidth;
			popupLeftAdjustment = adjustmentLeft;
		}
	}
</script>

<svelte:window bind:innerWidth={windowWidth} />
<span class="ref" bind:this={parent}>
	<input type="checkbox" id={checkbox_id} /><label for={checkbox_id}
		><span class="reftitle">[{title}]</span></label
	>
	<span
		class="refbody"
		style:left={`${popupLeftAdjustment}px`}
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
		min-width: 20em;
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
