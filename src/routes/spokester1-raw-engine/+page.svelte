<script>
	import { onMount } from "svelte";
	import * as THREE from "three";

	let container;
	let score = 0;
	let reward = 0;
	let showGameOver = false;
	let frameId;

	const Config = {
		physics: {
			RENDER_DISTANCE: 300.0,
			LANE_WIDTH: 2.5,
			PLAYER_LANE_SWITCH_SPEED: 10.0,
			INITIAL_WORLD_SPEED: 30.0,
			MAX_WORLD_SPEED: 250.0,
			WORLD_SPEEDUP_FACTOR: 4.0,
			DESPAWN_DISTANCE: -20.0,
			COLLISION_THRESH: 1.2,
		},
		player: {
			WIDTH: 0.5,
			HEIGHT: 1.0,
			LENGTH: 0.5,
		},
		car: {
			WIDTH: 1.9,
			HEIGHT: 1.7,
			LENGTH: 4.5,
		},
		building: {
			WIDTH: 10.0,
			HEIGHT: 15.0,
			LENGTH: 40.0,
			SPAWN_DIST: 15.0, // Distance from center
		},
		colors: {
			BG: 0x111111,
			PLAYER: 0x00ff00,
			CAR: 0xff0000,
			BUILDING: 0x555555,
		},
	};

	let game = {
		worldSpeed: Config.physics.INITIAL_WORLD_SPEED,
		worldDistance: 0,
		laneIdx: 1,
		lanes: [2.5, 0, -2.5],
		spawnRate: 0.01,
		spawnRateBuilding: 0.1,
		obstacles: [],
		scenery: [],
		over: false,
	};

	onMount(() => {
		const scene = new THREE.Scene();
		scene.background = new THREE.Color(Config.colors.BG);

		const camera = new THREE.PerspectiveCamera(
			30,
			container.clientWidth / container.clientHeight,
			0.1,
			1000,
		);
		const renderer = new THREE.WebGLRenderer({ antialias: true });
		renderer.setSize(container.clientWidth, container.clientHeight);
		container.appendChild(renderer.domElement);

		const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
		const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
		dirLight.position.set(5, 10, 7);
		scene.add(ambientLight, dirLight);

		const boxGeo = new THREE.BoxGeometry(1, 1, 1);

		// --- STABLE CHECKERBOARD SHADER ---
		const groundGeo = new THREE.PlaneGeometry(200, 2000);
		const groundMat = new THREE.ShaderMaterial({
			uniforms: {
				uOffset: { value: 0 },
				uScale: { value: 10.0 }, // Matches Pygame GROUND_CHECKER_SCALE
			},
			vertexShader: `
                varying vec2 vWorldCoord;
                uniform float uOffset;
                void main() {
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    // We pass the X and the scrolled Z to the fragment shader
                    vWorldCoord = vec2(worldPos.x, worldPos.z + uOffset);
                    gl_Position = projectionMatrix * viewMatrix * worldPos;
                }`,
			fragmentShader: `
                varying vec2 vWorldCoord;
                uniform float uScale;
                void main() {
                    vec2 pos = floor(vWorldCoord / uScale);
                    float pattern = mod(pos.x + pos.y, 2.0);
                    vec3 col = mix(vec3(0.15), vec3(0.25), pattern);
                    gl_FragColor = vec4(col, 1.0);
                }`,
		});
		const ground = new THREE.Mesh(groundGeo, groundMat);
		ground.rotation.x = -Math.PI / 2;
		scene.add(ground);

		// Player Mesh
		const player = new THREE.Group();
		const pBody = new THREE.Mesh(
			boxGeo,
			new THREE.MeshPhongMaterial({ color: Config.colors.PLAYER }),
		);
		pBody.scale.set(
			Config.player.WIDTH,
			Config.player.HEIGHT,
			Config.player.LENGTH,
		);
		pBody.position.y = Config.player.HEIGHT / 2;
		player.add(pBody);
		scene.add(player);

		// --- SPAWNING FUNCTIONS ---
		const spawnObstacle = () => {
			const x = game.lanes[Math.floor(Math.random() * 3)];
			const car = new THREE.Group();
			const mat = new THREE.MeshPhongMaterial({ color: Config.colors.CAR });

			const body = new THREE.Mesh(boxGeo, mat);
			body.scale.set(
				Config.car.WIDTH,
				Config.car.HEIGHT * 0.6,
				Config.car.LENGTH,
			);
			body.position.y = (Config.car.HEIGHT * 0.6) / 2;

			const cabin = new THREE.Mesh(boxGeo, mat);
			cabin.scale.set(
				Config.car.WIDTH * 0.8,
				Config.car.HEIGHT * 0.4,
				Config.car.LENGTH * 0.5,
			);
			cabin.position.y =
				Config.car.HEIGHT * 0.6 + (Config.car.HEIGHT * 0.4) / 2;
			cabin.position.z = -Config.car.LENGTH * 0.1;

			car.add(body, cabin);
			car.position.set(x, 0, Config.physics.RENDER_DISTANCE); // Spawn further out for smoothness
			game.obstacles.push(car);
			scene.add(car);
		};

		const spawnBuilding = () => {
			const x =
				Math.random() > 0.5
					? Config.building.SPAWN_DIST
					: -Config.building.SPAWN_DIST;
			const b = new THREE.Mesh(
				boxGeo,
				new THREE.MeshPhongMaterial({ color: Config.colors.BUILDING }),
			);
			b.scale.set(
				Config.building.WIDTH,
				Config.building.HEIGHT,
				Config.building.LENGTH,
			);
			b.position.set(
				x,
				Config.building.HEIGHT / 2,
				Config.physics.RENDER_DISTANCE,
			);
			game.scenery.push(b);
			scene.add(b);
		};

		const reset = () => {
			game.worldSpeed = Config.physics.INITIAL_WORLD_SPEED;
			game.worldDistance = 0;
			game.laneIdx = 1;
			score = 0;
			showGameOver = false;
			game.over = false;
			[...game.obstacles, ...game.scenery].forEach((o) => scene.remove(o));
			game.obstacles = [];
			game.scenery = [];
			player.position.set(0, 0, 0);
		};

		const handleKey = (e) => {
			if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
			if (game.over) {
				reset();
			}
			if (e.key === "ArrowLeft" && game.laneIdx > 0) game.laneIdx--;
			if (e.key === "ArrowRight" && game.laneIdx < 2) game.laneIdx++;
		};

		window.addEventListener("keydown", handleKey);

		const animate = () => {
			frameId = requestAnimationFrame(animate);
			const dt = 1 / 60;


			if (game.over) {
				player.position.z -= game.worldSpeed * dt;

				reward = 0;
			} else {
			    game.worldSpeed += Config.physics.WORLD_SPEEDUP_FACTOR * dt;
				reward = game.worldSpeed * dt;
				score += reward;

				const targetX = game.lanes[game.laneIdx];
				player.position.x +=
					(targetX - player.position.x) *
					Config.physics.PLAYER_LANE_SWITCH_SPEED *
					dt;
			}

            const worldTranslationZ = game.worldSpeed * dt;
            game.worldDistance += worldTranslationZ;


			groundMat.uniforms.uOffset.value = game.worldDistance;

			// Move and cleanup obstacles
			game.obstacles.forEach((obs, i) => {
				obs.position.z -= game.worldSpeed * dt;
				if (
					Math.abs(obs.position.z) < 2.0 &&
					Math.abs(obs.position.x - player.position.x) <
						Config.physics.COLLISION_THRESH
				) {
					game.over = true;
					showGameOver = true;
					// setTimeout(reset, 2000);
				}
				if (obs.position.z < Config.physics.DESPAWN_DISTANCE) {
					scene.remove(obs);
					game.obstacles.splice(i, 1);
				}
			});

			// Move and cleanup scenery
			game.scenery.forEach((b, i) => {
				b.position.z -= game.worldSpeed * dt;
				if (b.position.z < -50) {
					scene.remove(b);
					game.scenery.splice(i, 1);
				}
			});

			// Spawn logic
			if (Math.random() < game.spawnRate) spawnObstacle();
			if (Math.random() < game.spawnRateBuilding) spawnBuilding();

			// Camera behavior
			camera.position.lerp(
				new THREE.Vector3(player.position.x * 0.6, 2, -12),
				0.1,
			);
			camera.lookAt(player.position.x, 1, 20);

			renderer.render(scene, camera);
		};

		animate();

		return () => {
			window.removeEventListener("keydown", handleKey);
			cancelAnimationFrame(frameId);
			renderer.dispose();
		};
	});
</script>

<h1>Spokester1 - Raw Game Engine</h1>

<a href="spokester1-neural-game-graphics#game-engine">[&lt- Back to Spokester1 Article]</a>

<p>
	This is a JavaScript port of the Spokester1 game engine. The ground is a checkerboard pattern.
	There are cars to avoid and buildings on the side of the boulevard. There are
	no trees or textures or animations.
</p>

<p>
	This is what the game looks like without any neural graphics.
</p>

<p>
	Controls:<br/> Left Arrow: Move left <br/> Right Arrow: Move right
</p>

<div class="game-wrapper">
	<div bind:this={container} class="canvas-container">
		<div class="hud">
			<div>Score: {Math.floor(score)}</div>
			<div>Reward: {reward.toFixed(2)}</div>
		</div>

		{#if showGameOver}
			<div class="game-over">
				<h1>CRASHED</h1>
				<p>Press an arrow key to restart...</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.game-wrapper {
		width: 100%;
		height: 500px;
		background: #000;
		position: relative;
		border-radius: 4px;
		overflow: hidden;
	}
	.canvas-container {
		width: 100%;
		height: 100%;
	}
	.hud {
		position: absolute;
		top: 10px;
		left: 10px;
		color: #00ff00;
		font-family: monospace;
		font-size: 1.1rem;
		z-index: 5;
		text-shadow: 1px 1px 2px black;
	}
	.game-over {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		color: white;
		background: rgba(0, 0, 0, 0.8);
		padding: 20px;
		text-align: center;
		border-radius: 8px;
	}
</style>
