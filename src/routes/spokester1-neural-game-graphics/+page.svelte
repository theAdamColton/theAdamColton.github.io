<svelte:head>
	<title>Spokester 1: Neural Game Graphics</title>
</svelte:head>


<div id="spinningcube">
    <div style="transform: translate3d(0em, 0em, -1.5em)">Ever have that</div>
    <div style="transform: translate3d(0em, 0.1em, -0.5em)">feeling where</div>
    <div style="transform: translate3d(0em, 0.2em, 0em)">you're not sure</div>
    <div style="transform: translate3d(0em, 0.3em, 0.5em)">if you are awake</div>
    <div style="transform: translate3d(0em, 0.4em, 1em)">or dreaming?</div>
</div>


<style>
	#spinningcube {
		transform-style: preserve-3d;
		animation: spin 12s linear infinite;
		animation: spin 6s ease-in-out infinite alternate;
	}

	@keyframes spin {
		from { transform: rotateY(0deg); }
		to { transform: rotateY(360deg); }
		from { transform: rotateY(-20deg) rotateX(-5deg); }
		to { transform: rotateY(20deg) rotateX(5deg); }
	}
</style>