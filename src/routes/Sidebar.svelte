<script>
	export let animated = false;
</script>

<div class="sidebar">
	{#if animated}
		<img src="fire.webp" width="100%" height="20%" alt="fire" />
		<img src="fire.webp" width="100%" height="20%" alt="fire" />
		<img src="fire.webp" width="100%" height="20%" alt="fire" />
		<img src="fire.webp" width="100%" height="20%" alt="fire" />
		<img src="fire.webp" width="100%" height="20%" alt="fire" />
	{:else}
		<img src="firestatic.jpg" width="100%" height="20%" alt="fire" />
		<img src="firestatic.jpg" width="100%" height="20%" alt="fire" />
		<img src="firestatic.jpg" width="100%" height="20%" alt="fire" />
		<img src="firestatic.jpg" width="100%" height="20%" alt="fire" />
		<img src="firestatic.jpg" width="100%" height="20%" alt="fire" />
	{/if}
</div>
