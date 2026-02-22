package model;

import java.util.ArrayList;

import data.DataLoader;

public class HMM extends Model {
	
	/**
	 * TODO: Define any necessary variables here 
	 */
	
	public HMM() {
		super();
	}
	
	/**
	 * TODO: Complete the fit method
	 * Tip: probabilities are computed based on visit counts
	 * 		Update and use _stateVisitCounts, _state2StateVisitCounts, _obsVisitCounts arrays
	 */
	@Override
	public void fit(DataLoader trainLoader) {
				
		// Initialize the sizes of the visit count arrays and probability tables 
		initialize();
		
		//////////////////////////////////////////////////////////////
		/**
		 * TODO: Update the visit count arrays
		 */
		//////////////////////////////////////////////////////////////
		
		//////////////////////////////////////////////////////////////
		/** 
		 * TODO: Fit the transition CPT
		 * Tip: use _transitionProbability.setValue to set the value of the CPT
		 */
		for (State srcState : State.values()) {
			for (State destState : State.values()) {
				
			}
		}
		//////////////////////////////////////////////////////////////
		
		//////////////////////////////////////////////////////////////
		/**
		 * 
		 * TODO: Fit the emission CPT
		 * Tip: use _emissionProbability.setValue to set the value of the CPT
		 */
		//////////////////////////////////////////////////////////////
		
		// Re-normalize the probabilities to account for numerical errors
		_emissionProbability.normalize();
		_transitionProbability.normalize();
	}

	/**
	 * 	Decodes the latent states (i.e. finds out the most likely path of the states) 
	 * given a single document (i.e. a sequence of observations) via the Viterbi algorithm.
	 * 
	 * @param doc	The list of word tokens of a single document
	 * @return		An array of state IDs representing the most likely path of state progression
	 */
	@Override
	public Integer[] decode(ArrayList<String> doc) {
		Integer[] mostLikelyPath = null; 
		
		//////////////////////////////////////////////////////////////
		/**
		 * 
		 * TODO: Decode the sequence of the most likely state IDs given the list of word tokens
		 * 
		 * Note: the length of the returned sequence should be 'T + 2' if T is the total number 
		 * of word tokens in doc.
		 * 
		 */
		//////////////////////////////////////////////////////////////
		
		return mostLikelyPath;
	}
}
