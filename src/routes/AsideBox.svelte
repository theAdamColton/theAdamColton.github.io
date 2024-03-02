<script>
	// box for asides
	export const title = "";
</script>

<div class="aside-box-wrapper">
	<div class="aside-box-bk">
		<slot name="title" />
		<div class="aside-box">
			<div class="aside-box-item">
				<slot name="image" />
			</div>
			<div class="aside-box-item">
				<slot />
			</div>
		</div>
	</div>
</div>

<style>
	.aside-box {
		display: flex;
		align-items: center;
		flex-wrap:wrap;
	}
	.aside-box-bk {
		padding: 15px 15px;
		background-color: #acbabf;
	}
	.aside-box-wrapper {
		padding: 10px;
	}
	.aside-box-item {
		padding: 0px 10px;
	}
</style>
