#ifndef PCT_CONFIG_H
#define PCT_CONFIG_H

#define PROFILER   1
#define ITERATIONS 12	// # of iterations through the entire set of histories to perform in iterative image reconstruction
/***************************************************************************************************************************************************************************/
/********************************************************************* Execution and early exit options ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool RUN_ON                    = true;	// Turn preprocessing on/off (T/F) to enter individual function testing without commenting
const bool EXIT_AFTER_BINNING        = false;	// Exit program early after completing data read and initial processing
const bool EXIT_AFTER_HULLS          = false;	// Exit program early after completing hull-detection
const bool EXIT_AFTER_CUTS           = false;	// Exit program early after completing statistical cuts
const bool EXIT_AFTER_SINOGRAM       = false;	// Exit program early after completing the construction of the sinogram
const bool EXIT_AFTER_FBP            = true;	// Exit program early after completing FBP
/********************************************************************** Preprocessing option parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool DEBUG_TEXT_ON			 = true; 		// Provide (T) or suppress (F) print statements to console during execution
const bool SAMPLE_STD_DEV			 = true; 		// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
const bool FBP_ON					 = true; 		// Turn FBP on (T) or off (F)
const bool AVG_FILTER_FBP			 = false;	// Apply averaging filter to initial iterate (T) or not (F)
const bool MEDIAN_FILTER_FBP		 = false;
const bool IMPORT_FILTERED_FBP		 = false;
const bool SC_ON					 = false;	// Turn Space Carving on (T) or off (F)
const bool MSC_ON					 = true; 		// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON					 = false;	// Turn Space Modeling on (T) or off (F)
const bool AVG_FILTER_HULL			 = true; 		// Apply averaging filter to hull (T) or not (F)
const bool COUNT_0_WEPLS			 = false;	// Count the number of histories with WEPL = 0 (T) or not (F)
const bool REALLOCATE				 = false;
const bool MLP_FILE_EXISTS			 = false;
const bool MLP_ENDPOINTS_FILE_EXISTS = true;
bool MODIFY_MLP = true;
/***************************************************************************************************************************************************************************/
/***************************************************************** Input/output specifications and options *****************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/******************************************************************* Path to the input/output directories ******************************************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_DIRECTORY[]   = "//home//karbasi//Public//";
const char OUTPUT_DIRECTORY[]  = "//home//karbasi//Public//";
/***************************************************************************************************************************************************************************/
/******************************************** Name of the folder where the input data resides and output data is to be written *********************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_FOLDER[]  = "input_CTP404_4M";
const char OUTPUT_FOLDER[] = "cuda_test_paniz";


#endif /* PCT_CONFIG_H */
