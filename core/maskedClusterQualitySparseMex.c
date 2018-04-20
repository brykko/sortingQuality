#include "mex.h"
#include "matrix.h"
#include <stdint.h>

#define DEBUG 0

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/*
	 * Usage:
	 * [fet_other, nspikes] = maskedCQSMex(nChan, fet, clus, these_chans, clu_ids, c, fet_inds)
	 */

	int fet_n_chans = mxGetScalar(prhs[0]);

	double *fet = mxGetPr(prhs[1]);
	mwSize *dims = mxGetDimensions(prhs[1]);
	size_t n_spikes = (size_t)dims[0];
	size_t n_fet_per_chan = (size_t)dims[1];

	size_t n_clus = mxGetM(prhs[4]);

	int *clu = (int *)mxGetData(prhs[2]);
	int16_t *these_chans = (int16_t *)mxGetData(prhs[3]);
	int32_t *clu_ids = (int32_t *)mxGetData(prhs[4]);
	int16_t *fet_inds = (int16_t *)mxGetData(prhs[6]);

	int c = mxGetScalar(prhs[5]);

	dims = mxCalloc(3, sizeof(mwSize));
	dims[0] = n_spikes;
	dims[1] = n_fet_per_chan;
	dims[2] = fet_n_chans;

	plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	mxFree(dims);

	double *fet_other = mxGetPr(plhs[0]);

	int16_t *chans_c2_has = mxCalloc(fet_n_chans, sizeof(int16_t));

	int clu_first_spike;

	size_t c_spikes = 0;
	size_t s = 0;
	size_t s_thisclu_first = 0;
	size_t s_thisclu_last = 0;
	size_t c_spikes_first = 0;
	size_t c_spikes_last = 0;

	int16_t *idx_chan_match = &(int16_t){ 0 };

	for (int c2 = 0; c2 < n_clus; c2++) {

		if (c2 != c) {

			// Advance to spikes from the queried cluster
			while (s < n_spikes && clu[s] < clu_ids[c2]) { s++; }

			if (DEBUG) { mexPrintf("Starting query clu%u at spike idx %u\n", c2, s); }

			for (int i = 0; i < fet_n_chans; i++) {
				int idx = c2 + i*n_clus;
				chans_c2_has[i] = fet_inds[idx];
			}

			// Record the first spike index for this cluster
			s_thisclu_first = s;
			c_spikes_first = c_spikes;

			for (int f = 0; f < fet_n_chans; f++) {

				//mexPrintf("ismember(%u, [%u, %u, %u, %u])\n", these_chans[f], chans_c2_has[0], chans_c2_has[1], chans_c2_has[2], chans_c2_has[3]);

				if (any_ismember(these_chans + f, 1, chans_c2_has, fet_n_chans, idx_chan_match)) {

					if (DEBUG) {
						clu_first_spike = 1;
						mexPrintf("First spike fet %u , fet_idx = %u: ", f, *idx_chan_match);
					}

					while (s < n_spikes && clu[s] == clu_ids[c2]) {

						// For every spike in the queried cluster, extract its features
						for (int ch = 0; ch < n_fet_per_chan; ch++) {
							int ir = s;
							int ic = n_spikes*ch;
							int id = n_spikes*n_fet_per_chan*(*idx_chan_match);

							int id_o = n_spikes*n_fet_per_chan*f;
							int ir_o = c_spikes;
							fet_other[ir_o + ic + id_o] = fet[ir + ic + id];

							if (DEBUG) { if (clu_first_spike){ mexPrintf("ch%u idx=%u, val=%.3f ", ch, ir + ic + id, fet[ir + ic + id]); } }
						}

						if (DEBUG) {
							// Move on to the next spike
							if (clu_first_spike) {
								mexPrintf("\n");
							}
							clu_first_spike = 0;
						}
						c_spikes++;
						s++;
					}

					if (DEBUG) { mexPrintf("Appended clu %u chan %u spikes %u-%u fet idx %u, total spikes %u\n", c2, these_chans[f], s_thisclu_first, s - 1, *idx_chan_match, c_spikes); }

					// Cycling to a new feature: reset the spike indices to the first spike for the current cluster 
					s_thisclu_last = s;
					c_spikes_last = c_spikes;
					s = s_thisclu_first;
					c_spikes = c_spikes_first;

					if (DEBUG) { mexPrintf("Reset c_spikes to start of clu: from %u to %u\n", c_spikes_last, c_spikes); }
				}
			}

			// Finished looping through all features for this cluster:
			// Reset the spike inds to the beginning of the next cluster
			if (DEBUG) { mexPrintf("Reset c_spikes for next clu: from %u to %u\n\n", c_spikes, c_spikes_last); }
			c_spikes = c_spikes_last;
			s = s_thisclu_last;

		}
	}

	plhs[1] = mxCreateDoubleScalar(c_spikes);
	mxFree(chans_c2_has);
	return 0;

}

int any_ismember(int16_t* a, int na, int16_t* b, int nb, int16_t *idx_b_match) {
	for (int i = 0; i < na; i++) {
		for (int j = 0; j < nb; j++) {
			if (a[i] == b[j]) {
				*idx_b_match = (int16_t)j;
				return 1;
			}
		}
	}
	return 0;
}