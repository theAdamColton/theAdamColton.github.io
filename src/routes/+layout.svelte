<script>
	import Header from './Header.svelte';
	import Footer from './Footer.svelte';
	import './styles.css';
	import Sidebar from './Sidebar.svelte';
</script>

<div class="app">
	<Header />





	<div class="sidebarcontainer">
		<Sidebar/>
		<div class="pagecontent">
			<main>
				<slot />
			</main>


		<hr />

		<footer>
		<Footer/>
		</footer>

		</div>

		<Sidebar/>
	</div>

</div>
