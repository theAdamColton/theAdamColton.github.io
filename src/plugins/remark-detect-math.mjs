import { visit } from "unist-util-visit";

// Flags posts that contain math so the layout only loads the KaTeX css there.
export default function remarkDetectMath() {
	return (tree, file) => {
		visit(tree, ["math", "inlineMath"], () => {
			file.data.astro.frontmatter.math = true;
		});
	};
}
