import adapter from '@sveltejs/adapter-static';


export default {
	kit: {
		adapter: adapter({
			// default options are shown. On some platforms
			// these options are set automatically — see below
			pages: 'build',
			assets: 'build',
			fallback: undefined,
			precompress: false,
			strict: true
		}),
	prerender: {
			handleHttpError: ({ path, referrer, message }) => {
				if (referrer === '/a-picture-is-worth-8x8x8-words') {
					return;
				}

				// otherwise fail the build
				throw new Error(message);
			}
		}
	}
};
