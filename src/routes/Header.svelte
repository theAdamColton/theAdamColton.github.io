<script>
	export let ishome = true;
</script>

{#if !ishome}
	<h1>
		<a href="/">home</a>
	</h1>
{:else}
	<div class="rainbow-text-animated main-title">
		Adam Colton: University of Utah 2023
	</div>
{/if}
