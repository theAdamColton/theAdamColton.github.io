<script>
	import Header from "./Header.svelte";
	import Footer from "./Footer.svelte";
	import "./styles.css";
	import Sidebar from "./Sidebar.svelte";

	import { page } from "$app/stores";
	$: ishome = $page.route.id === "/";
</script>

<div class="app">
	<div class="sidebarcontainer">
		<Sidebar animated={ishome} />
		<div class="pagecontent">
			<Header {ishome} />
			<main>
				<slot />
			</main>

			<hr />

			<footer>
				<Footer />
			</footer>
		</div>

		<Sidebar animated={ishome} />
	</div>
</div>
