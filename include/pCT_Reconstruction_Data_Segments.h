//#ifndef PCT_RECONSTRUCTION_H
//#define PCT_RECONSTRUCTION_H
#pragma once
/***************************************************************************************************************************************************************************/
/********************************************************************* Header for pCT reconstruction program ***************************************************************/
/***************************************************************************************************************************************************************************/
//#include <Windows4Root.h>
//#include "w32pragma.h"
//#include <TROOT.h>
//#include <TMath.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>


#include <algorithm>    // std::transform
//#include <array>
#include <cmath>
#include <cstdarg>		// va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// rand, srand
#include <cstring>
#include <ctime>		// clock(), time() 
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <new>			 
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
#include <omp.h>		// OpenMP
#include <sstream>
#include <string>
//#include <tuple>
#include <typeinfo>		//operator typeid
//#include <type_traits>	// is_pod
#include <utility> // for std::move
#include <vector>

//using namespace std::placeholders; 
//using namespace std;
using std::cout;
using std::endl;
typedef unsigned long long ULL;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************ Preprocessing usage options **********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

struct generic_input_container
{
	char* key;
	unsigned int input_type_ID;
	int integer_input;		// type_ID = 1
	double double_input;	// type_ID = 2
	char string_input[512];	// type_ID = 3
};

// Container for all config file specified parameters allowing these to be transffered to GPU with single statements
// 8 UI, 18D, 6 C*
struct parameters
{
	double lambda;
	//	char INPUT_DIRECTORY_D[256];
	//	char OUTPUT_DIRECTORY_D[256];
	//	char INPUT_FOLDER_D[256];
	//	char OUTPUT_FOLDER_D[256];
	//	char INPUT_BASE_NAME_D[32];
	//	char FILE_EXTENSION_D[4];
	//	unsigned int GANTRY_ANGLES_D;
	//	unsigned int NUM_SCANS_D;
	//	unsigned int T_BINS_D;
	//	unsigned int V_BINS_D;
	//	unsigned int COLUMNS_D;
	//	unsigned int ROWS_D;
	//	unsigned int SLICES_D;
	//	unsigned int SIGMAS_TO_KEEP_D;
	//	double SSD_T_SIZE_D;
	//	double SSD_V_SIZE_D;
	//	double T_SHIFT_D;
	//	double U_SHIFT_D;
	//	double T_BIN_SIZE_D;	
	//	double V_BIN_SIZE_D;	
	//	double ANGULAR_BIN_SIZE_D;	
	//	double RECON_CYL_RADIUS_D;
	//	double RECON_CYL_HEIGHT_D;
	//	double IMAGE_WIDTH_D;
	//	double IMAGE_HEIGHT_D;
	//	double IMAGE_THICKNESS_D;
	//	double VOXEL_WIDTH_D;
	//	double VOXEL_HEIGHT_D;
	//	double VOXEL_THICKNESS_D;
	//	double lambda_D;
	//	double LAMBDA_D;
	//	double parameter_D;
};
parameters parameter_container;
parameters *parameters_h = &parameter_container;
parameters *parameters_d;

std::map<std::string,unsigned int> switchmap;
unsigned int num_run_arguments;
char** run_arguments;
char* CONFIG_DIRECTORY;
char* input_directory;
double lambda;
int parameter;
int num_voxel_scales;
double* voxel_scales;

// CTP_404
// 20367505: 32955315 -> 32955313									// No average filtering on hull
// 20367499: 32955306 ->32955301									// No average filtering on hull
// 20573129: 5143282->5143291										// r=1
// 20648251: 33409572->33409567										// r=2
// 20648257: 33409582 -> 33409577/33409603//5162071;				// r=2
// 20764061: 33596956->33596939/33596977							// r=3 

// CTP_404M
// 20153778: 5038452-> 5038457										// r=1
//ULL NUM_RECON_HISTORIES = 20574733
ULL NUM_RECON_HISTORIES =105642524;
ULL PRIME_OFFSET = 26410633;
/***************************************************************************************************************************************************************************/
/********************************************************************* Execution and early exit options ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool RUN_ON				= true;									// Turn preprocessing on/off (T/F) to enter individual function testing without commenting
const bool EXIT_AFTER_BINNING	= false;									// Exit program early after completing data read and initial processing
const bool EXIT_AFTER_HULLS		= false;									// Exit program early after completing hull-detection
const bool EXIT_AFTER_CUTS		= false;									// Exit program early after completing statistical cuts
const bool EXIT_AFTER_SINOGRAM	= false;									// Exit program early after completing the construction of the sinogram
const bool EXIT_AFTER_FBP		= false;									// Exit program early after completing FBP
/***************************************************************************************************************************************************************************/
/********************************************************************** Preprocessing option parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool DEBUG_TEXT_ON				= true;								// Provide (T) or suppress (F) print statements to console during execution
const bool SAMPLE_STD_DEV				= true;								// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
const bool FBP_ON						= true;								// Turn FBP on (T) or off (F)
const bool AVG_FILTER_FBP				= false;							// Apply averaging filter to initial iterate (T) or not (F)
const bool MEDIAN_FILTER_FBP			= false; 
const bool IMPORT_FILTERED_FBP			= false;
const bool SC_ON						= false;							// Turn Space Carving on (T) or off (F)
const bool MSC_ON						= true;								// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON						= false;							// Turn Space Modeling on (T) or off (F)
const bool AVG_FILTER_HULL				= true;								// Apply averaging filter to hull (T) or not (F)
const bool COUNT_0_WEPLS				= false;							// Count the number of histories with WEPL = 0 (T) or not (F)
const bool REALLOCATE					= false;
const bool MLP_FILE_EXISTS				= false;
const bool MLP_ENDPOINTS_FILE_EXISTS	= true;
bool MODIFY_MLP = false;
/***************************************************************************************************************************************************************************/
/***************************************************************** Input/output specifications and options *****************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/******************************************************************* Path to the input/output directories ******************************************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_DIRECTORY[]   = "//home//share//pCT_data//";
const char OUTPUT_DIRECTORY[]  = "//home//share//pCT_data//";
/***************************************************************************************************************************************************************************/
/******************************************** Name of the folder where the input data resides and output data is to be written *********************************************/
/***************************************************************************************************************************************************************************/
//const char INPUT_FOLDER[]	   = "merged_data";
//const char OUTPUT_FOLDER[]	   = "output_data";
//const char INPUT_FOLDER[]	   = "merged_data";
//const char OUTPUT_FOLDER[]	   = "merged_data";
//const char INPUT_FOLDER[]	   = "beam-Sep2014\\bin-Sensitom";
//const char OUTPUT_FOLDER[]	   = "beam-Sep2014\\bin-Sensitom";
//const char INPUT_FOLDER[]	   = "beam-Sep2014\\bin-CIRSFordInf";
//const char OUTPUT_FOLDER[]	   = "beam-Sep2014\\bin-CIRSFordInf";
//const char INPUT_FOLDER[]	   = "beam-Sep2014\\bin-CIRSFordSup";
//const char OUTPUT_FOLDER[]	   = "beam-Sep2014\\bin-CIRSFordSup";
const char INPUT_FOLDER[]	   = "input_CTP404_4M";
const char OUTPUT_FOLDER[]         = "cuda_test";
//const char INPUT_FOLDER[]	   = "output_HeadPhantom";
//const char OUTPUT_FOLDER[]	   = "output_HeadPhantom";
//const char INPUT_FOLDER[]	   = "PedHead-july";
//const char OUTPUT_FOLDER[]	   = "PedHead-july";
//const char INPUT_FOLDER[]	   = "output_ESFPhant";
//const char OUTPUT_FOLDER[]	   = "output_ESFPhant";
//const char INPUT_FOLDER[]	   = "input_CTP404";
//const char OUTPUT_FOLDER[]	   = "input_CTP404";
//const char INPUT_FOLDER[]	   = "input_water_GeantNONUC";
//const char OUTPUT_FOLDER[]	   = "input_water_GeantNONUC";
//const char INPUT_FOLDER[]	   = "input_water_Geant500000";
//const char OUTPUT_FOLDER[]   = "input_water_Geant500000";
//const char INPUT_FOLDER[]	   = "waterPhantom";
//const char OUTPUT_FOLDER[]   = "waterPhantom";
//const char INPUT_FOLDER[]	   = "catphan";
//const char OUTPUT_FOLDER[]   = "catphan";
//const char INPUT_FOLDER[]	   = "DetectData";
//const char OUTPUT_FOLDER[]   = "DetectData";
//const char INPUT_FOLDER[]	   = "Rat_Scan2";
//const char OUTPUT_FOLDER[]   = "Rat_Scan2"; 
//const char INPUT_FOLDER[]	   = "DetectDataWeplNoisy1";
//const char OUTPUT_FOLDER[]   = "DetectDataWeplNoisy1";
//const char INPUT_FOLDER[]	   = "input_noisefloor40";
//const char OUTPUT_FOLDER[]   = "input_noisefloor40";
/***************************************************************************************************************************************************************************/
/**************************************** Prefix of the input data set filename (_trans%d_%03d.txt (or .dat) will be added to this) ****************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_BASE_NAME[]   = "projection";							// waterPhantom, catphan, input_water_Geant500000
//const char INPUT_BASE_NAME[] = "simdata";								// Simulated data sets generated by Micah: DetectData, DetectDataWeplNoisy1, NoisyUniform1,...
//const char INPUT_BASE_NAME[] = "rat_scan2_shift";						// Anesthetized rat held in restraints
//const char INPUT_BASE_NAME[] = "ped_scan1";							// Anthropomorphic pediatric head phantom (Model 715-HN, CIRS1)
/***************************************************************************************************************************************************************************/
/******************************************************************** File extension for the input data ********************************************************************/
/***************************************************************************************************************************************************************************/
const char FILE_EXTENSION[]	   = ".bin";								// Binary file extension
//const char FILE_EXTENSION[]  = ".dat";								// Generic data file extension, independent of encoding (various encodings can be used)
//const char FILE_EXTENSION[]  = ".txt";								// ASCII text file extension
/***************************************************************************************************************************************************************************/
/****************************************************************** Input data specification parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
enum DATA_FORMATS { OLD_FORMAT, VERSION_0, VERSION_1 };					// Define the data formats that are supported
const DATA_FORMATS				DATA_FORMAT = VERSION_0;				// Specify which data format to use for this run
unsigned int PHANTOM_NAME_SIZE;
unsigned int DATA_SOURCE_SIZE;
unsigned int PREPARED_BY_SIZE;
unsigned int MAGIC_NUMBER_CHECK = int('DTCP');
unsigned int SKIP_2_DATA_SIZE;
unsigned int VERSION_ID;
unsigned int PROJECTION_INTERVAL;
const bool BINARY_ENCODING	   = true;									// Input data provided in binary (T) encoded files or ASCI text files (F)
const bool SINGLE_DATA_FILE    = false;									// Individual file for each gantry angle (T) or single data file for all data (F)
const bool SSD_IN_MM		   = true;									// SSD distances from rotation axis given in mm (T) or cm (F)
const bool DATA_IN_MM		   = true;									// Input data given in mm (T) or cm (F)
const bool MICAH_SIM		   = false;									// Specify whether the input data is from Micah's simulator (T) or not (F)
/***************************************************************************************************************************************************************************/
/**************************************************************************** Output filenames *****************************************************************************/
/***************************************************************************************************************************************************************************/
const char SC_HULL_FILENAME[]		= "x_SC";
const char MSC_HULL_FILENAME[]		= "x_MSC";
const char SM_HULL_FILENAME[]		= "x_SM";
const char FBP_HULL_FILENAME[]		= "x_FBP";
const char FBP_IMAGE_FILENAME[]		= "FBP_image";
const char* MLP_PATHS_FILENAME		= "MLP_paths";
const char* MLP_PATHS_ERROR_FILENAME    = "MLP_path_error";
const char* MLP_ENDPOINTS_FILENAME	= "MLP_endpoints";
const char* INPUT_ITERATE_FILENAME	= "FBP_med7.bin";
const char* IMPORT_FBP_FILENAME		= "FBP_med";
const char* INPUT_HULL_FILENAME		= "input_hull.bin";
char IMPORT_FBP_PATH[256];
char INPUT_ITERATE_PATH[256];
/***************************************************************************************************************************************************************************/
/************************************************************************ Output option parameters *************************************************************************/
/***************************************************************************************************************************************************************************/
const bool WRITE_SC_HULL		= true;								// Write SC hull to disk (T) or not (F)
const bool WRITE_MSC_COUNTS		= true;								// Write MSC counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_MSC_HULL		= true;								// Write MSC hull to disk (T) or not (F)
const bool WRITE_SM_COUNTS		= true;								// Write SM counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_SM_HULL		= true;								// Write SM hull to disk (T) or not (F)
const bool WRITE_FBP_IMAGE		= true;								// Write FBP image before thresholding to disk (T) or not (F)
const bool WRITE_FBP_HULL		= true;								// Write FBP hull to disk (T) or not (F)
const bool WRITE_AVG_FBP		= true;								// Write average filtered FBP image before thresholding to disk (T) or not (F)
const bool WRITE_MEDIAN_FBP		= false;							// Write median filtered FBP image to disk (T) or not (F)
const bool WRITE_FILTERED_HULL		= true;								// Write average filtered FBP image to disk (T) or not (F)
const bool WRITE_X_HULL			= true;								// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_0			= true;								// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_KI			= true;								// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X				= false;							// Write the reconstructed image to disk (T) or not (F)
/***************************************************************************************************************************************************************************/
/*************************************************************** Binned data analysis options and parameters ***************************************************************/
/***************************************************************************************************************************************************************************/
enum BIN_ANALYSIS_TYPE { MEANS, COUNTS, MEMBERS };							// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR { ALL_BINS, SPECIFIC_BINS };							// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION { BY_BIN, BY_HISTORY };								// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF { WEPLS, ANGLES, POSITIONS, BIN_NUMS };						// Choices for which type of binned data is desired
const bool WRITE_BIN_WEPLS	   = false;								// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
const bool WRITE_WEPL_DISTS	   = false;								// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
const bool WRITE_SSD_ANGLES    = false;									// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************* Preprocessing Constants *************************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************************************* Host/GPU computation and structure information **************************************************************/
/***************************************************************************************************************************************************************************/
#define BYTES_PER_HISTORY		48								// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define MAX_GPU_HISTORIES		static_cast<int>(3200000)		// [#] Number of histories to process on the GPU at a time, based on GPU capacity
#define MAX_CUTS_HISTORIES		static_cast<int>(3200000)
								// [#] Number of threads assigned to each block on the GPU
/***************************************************************************************************************************************************************************/
/**************************************** Scanning and detector system	(source distance, tracking plane dimensions) parameters ********************************************/
/***************************************************************************************************************************************************************************/
#define SOURCE_RADIUS			260.7								// [cm] Distance  to source/scatterer 
#define GANTRY_ANGLE_INTERVAL	4.0									// [degrees] Angle between successive projection angles 
#define GANTRY_ANGLES			int( 360 / GANTRY_ANGLE_INTERVAL )				// [#] Total number of projection angles
#define NUM_SCANS				1							// [#] Total number of scans
#define NUM_FILES				( NUM_SCANS * GANTRY_ANGLES )				// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE				35.0							// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE				9.0							// [cm] Length of SSD in v (vertical) direction
/***************************************************************************************************************************************************************************/
/************************************************* Binning (for statistical analysis) and sinogram (for FBP) parameters ****************************************************/
/***************************************************************************************************************************************************************************/
#define T_SHIFT					0.0							// [cm] Amount by which to shift all t coordinates on input
#define U_SHIFT					0.0							// [cm] Amount by which to shift all u coordinates on input
#define V_SHIFT					0.0							// [cm] Amount by which to shift all v coordinates on input
//#define T_SHIFT				   2.05							// [cm] Amount by which to shift all t coordinates on input
//#define U_SHIFT				   -0.16						// [cm] Amount by which to shift all u coordinates on input
#define T_BIN_SIZE				0.1							// [cm] Distance between adjacent bins in t (lateral) direction
#define T_BINS					int( SSD_T_SIZE / T_BIN_SIZE + 0.5 )			// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
#define V_BIN_SIZE				0.25							// [cm] Distance between adjacent bins in v (vertical) direction
#define V_BINS					int( SSD_V_SIZE/ V_BIN_SIZE + 0.5 )			// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
#define ANGULAR_BIN_SIZE		4.0								// [degrees] Angle between adjacent bins in angular (rotation) direction
#define ANGULAR_BINS			int( 360 / ANGULAR_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for path angle 
#define NUM_BINS				( ANGULAR_BINS * T_BINS * V_BINS )		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
#define SIGMAS_TO_KEEP			3										// [#] Number of standard deviations from mean to allow before cutting the history 
enum FILTER_TYPES {RAM_LAK, SHEPP_LOGAN, NONE};							// Define the types of filters that are available for use in FBP
const FILTER_TYPES				FBP_FILTER = SHEPP_LOGAN;			  	// Specifies which of the defined filters will be used in FBP
#define RAM_LAK_TAU				2/ROOT_TWO * T_BIN_SIZE					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from FBP_image
const unsigned int FBP_AVG_RADIUS = 1;
const double FBP_AVG_THRESHOLD = 0.1;
const unsigned int FBP_MEDIAN_RADIUS = 3;
/***************************************************************************************************************************************************************************/
/******************************************************************* Reconstruction cylinder parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
#define RECON_CYL_RADIUS		10.0									// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER		( 2 * RECON_CYL_RADIUS )				// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT		(SSD_V_SIZE - 1.0)						// [cm] Height of reconstruction cylinder
/***************************************************************************************************************************************************************************/
/********************************************************************	Reconstruction image parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
#define IMAGE_WIDTH				RECON_CYL_DIAMETER						// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT			RECON_CYL_DIAMETER						// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS			( SLICES * SLICE_THICKNESS )			// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define COLUMNS					200										// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS					200										// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICES					int( RECON_CYL_HEIGHT / SLICE_THICKNESS)// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS				( COLUMNS * ROWS * SLICES )				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define VOXEL_WIDTH				( RECON_CYL_DIAMETER / COLUMNS )		// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT			( RECON_CYL_DIAMETER / ROWS )			// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS			( IMAGE_THICKNESS / SLICES )			// [cm] Distance between top and bottom of each slice in image
#define SLICE_THICKNESS			0.25									// [cm] Distance between top and bottom of each slice in image
#define X_ZERO_COORDINATE		-RECON_CYL_RADIUS						// [cm] x-coordinate corresponding to front edge of 1st voxel (i.e. column) in image space
#define Y_ZERO_COORDINATE		RECON_CYL_RADIUS						// [cm] y-coordinate corresponding to front edge of 1st voxel (i.e. row) in image space
#define Z_ZERO_COORDINATE		RECON_CYL_HEIGHT/2						// [cm] z-coordinate corresponding to front edge of 1st voxel (i.e. slice) in image space
#define X_INCREASING_DIRECTION	RIGHT									// [#] specifies direction (LEFT/RIGHT) along x-axis in which voxel #s increase
#define Y_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along y-axis in which voxel #s increase
#define Z_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along z-axis in which voxel #s increase
/***************************************************************************************************************************************************************************/
/************************************************************************ Hull-Detection Parameters ************************************************************************/
/***************************************************************************************************************************************************************************/
#define MSC_DIFF_THRESH			50										// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SC_THRESHOLD			0.0										// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
#define MSC_THRESHOLD			0.0									// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
#define SM_LOWER_THRESHOLD		6.0										// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD		21.0									// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD		1.0										// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
#define HULL_FILTER_THRESHOLD	0.1										// [#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
#define HULL_FILTER_RADIUS		1										// [#] Averaging filter neighborhood radius in: [voxel - AVG_FILTER_SIZE, voxel + AVG_FILTER_RADIUS]
/***************************************************************************************************************************************************************************/
/****************************************************************************** MLP Parameters *****************************************************************************/
/***************************************************************************************************************************************************************************/
enum  HULL_TYPES {SC_HULL, MSC_HULL, SM_HULL, FBP_HULL };				// Define valid choices for which hull to use in MLP calculations
const HULL_TYPES				MLP_HULL = MSC_HULL;					// Specify which of the HULL_TYPES to use in this run's MLP calculations
#define E_0						13.6									// [MeV/c] empirical constant
#define X0						36.08									// [cm] radiation length
#define RSP_AIR					0.00113									// [cm/cm] Approximate RSP of air
#define VOXEL_STEP_SIZE			( VOXEL_WIDTH / 2 )						// [cm] Length of the step taken along the path, i.e. change in depth per step for
#define MAX_INTERSECTIONS		512										// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
#define MLP_U_STEP				( VOXEL_WIDTH / 2)						// Size of the step taken along u direction during MLP; depth difference between successive MLP points
const int max_path_elements = int(sqrt(double( ROWS^2 + COLUMNS^2 + SLICES^2)));

// Coefficients of 5th order polynomial fit to the term [1 / ( beta^2(u)*p^2(u) )] present in scattering covariance matrices Sigma 1/2 for:
#define BEAM_ENERGY				200
// 200 MeV protons
#define A_0						7.457  * pow( (float)10, (float)-6.0  )
#define A_1						4.548  * pow( (float)10, (float)-7.0  )
#define A_2						-5.777 * pow( (float)10, (float)-8.0  )
#define A_3						1.301  * pow( (float)10, (float)-8.0  )
#define A_4						-9.228 * pow( (float)10, (float)-10.0 )
#define A_5						2.687  * pow( (float)10, (float)-11.0 )

//// 200 MeV protons
//#define A_0						202.20574
//#define A_1						-7.6174839
//#define A_2						0.9413194
//#define A_3						-0.1141406
//#define A_4						0.0055340
//#define A_5						-0.0000972

// Common fractional values of A_i coefficients appearing in polynomial expansions of MLP calculations, precalculating saves time
#define A_0_OVER_2				A_0/2 
#define A_0_OVER_3				A_0/3
#define A_1_OVER_2				A_1/2
#define A_1_OVER_3				A_1/3
#define A_1_OVER_4				A_1/4
#define A_1_OVER_6				A_1/6
#define A_1_OVER_12				A_1/12
#define A_2_OVER_3				A_2/3
#define A_2_OVER_4				A_2/4
#define A_2_OVER_5				A_2/5
#define A_2_OVER_12				A_2/12
#define A_2_OVER_30				A_2/30
#define A_3_OVER_4				A_3/4
#define A_3_OVER_5				A_3/5
#define A_3_OVER_6				A_3/6
#define A_3_OVER_20				A_3/20
#define A_3_OVER_60				A_3/60
#define A_4_OVER_5				A_4/5
#define A_4_OVER_6				A_4/6
#define A_4_OVER_7				A_4/7
#define A_4_OVER_30				A_4/30
#define A_4_OVER_105			A_4/105
#define A_5_OVER_6				A_5/6
#define A_5_OVER_7				A_5/7
#define A_5_OVER_8				A_5/8
#define A_5_OVER_42				A_5/42
#define A_5_OVER_168			A_5/168
/***************************************************************************************************************************************************************************/
/*************************************************************** Iterative Image Reconstruction Parameters *****************************************************************/
/***************************************************************************************************************************************************************************/
const bool AVG_FILTER_ITERATE	= false;									// Apply averaging filter to initial iterate (T) or not (F)
const bool DIRECT_IMAGE_RECONSTRUCTION = false;

const unsigned int ITERATE_FILTER_RADIUS = 3;
const double ITERATE_FILTER_THRESHOLD = 0.1;
enum  INITIAL_ITERATE { X_HULL, FBP_IMAGE, HYBRID, ZEROS, IMPORT };		// Define valid choices for which hull to use in MLP calculations
const INITIAL_ITERATE			X_0 = HYBRID;							// Specify which of the HULL_TYPES to use in this run's MLP calculations

#define DECAY_FACTOR			(0.99 / pow(RECON_CYL_RADIUS, 2.0 ))	// Defines "a" in the lambda scaling factor 1 - a*r(i)^2 used to generate a radially dependent lambda
#define EXPONENTIAL_DECAY		(5/RECON_CYL_RADIUS)					// Defines "a" in the lambda scaling factor exp(-a*r) used to generate a radially dependent lambda
#define EXPONENTIAL_SQD_DECAY	(0.5/RECON_CYL_RADIUS)					// Defines "a" in the lambda scaling factor exp(-a*r^2) used to generate a radially dependent lambda
//#define EXPONENTIAL_TERM	exp(-0.5/RECON_CYL_RADIUS)					// Defines "a" in the lambda scaling factor exp(-a*r^2) used to generate a radially dependent lambda
#define AFFECT_RADIUS_SQD		pow(6.0, 2.0)
enum PROJECTION_ALGORITHMS { ART, SART, DROP, BIP, SAP };				// Define valid choices for iterative projection algorithm to use
const PROJECTION_ALGORITHMS		PROJECTION_ALGORITHM = DROP;			// Specify which of the projection algorithms to use for image reconstruction
unsigned int reconstruction_histories = 0;
//#define LAMBDA					0.001								// Relaxation parameter to use in image iterative projection reconstruction algorithms					
double ETA                                      = 2.5;
unsigned int METHOD                             = 1;
int PSI_SIGN                           = 1;
double* norm_Ai;
//#define LAMBDA					0.0001	
float LAMBDA = 0.001;// Relaxation parameter to use in image iterative projection reconstruction algorithms	
#define ITERATIONS				12										// # of iterations through the entire set of histories to perform in iterative image reconstruction

//#define BLOCK_SIZE			static_cast<int>(320000)//15000000//3840// # of paths to use for each update: ART = 1, 
//#define BLOCK_SIZE_RECON		static_cast<int>(3200)//3840// # of paths to use for each update: ART = 1, 
#define DROP_BLOCK_SIZE			static_cast<int>(320000)//3840// # of paths to use for each update: ART = 1, 

#define CONSTANT_CHORD_NORM		pow(VOXEL_WIDTH, 2.0)
double CONSTANT_LAMBDA_SCALE =	VOXEL_WIDTH * LAMBDA;

#define THREADS_PER_BLOCK		static_cast<int>(320)
#define ENDPOINTS_PER_BLOCK 		static_cast<int>(320)
#define HISTORIES_PER_BLOCK 		static_cast<int>(320)

#define ENDPOINTS_PER_THREAD 		static_cast<int>(1)
#define HISTORIES_PER_THREAD 		static_cast<int>(1)
#define VOXELS_PER_THREAD 		static_cast<int>(1)

#define MAX_ENDPOINTS_HISTORIES 	static_cast<int>(64000000)
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------- Timers ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
clock_t begin_endpoints, end_endpoints, clock_cycles_endpoints;
clock_t begin_init_image, end_init_image, clock_cycles_init_image;
clock_t begin_tables, end_tables, clock_cycles_tables;
clock_t begin_DROP_iteration, end_DROP_iteration, clock_cycles_DROP_iteration;
clock_t begin_DROP, end_DROP, clock_cycles_DROP;
clock_t begin_update_calcs, end_update_calcs, clock_cycles_update_calcs;
clock_t begin_update_image, end_update_image, clock_cycles_update_image;
clock_t begin_preprocessing, end_preprocessing, clock_cycles_preprocessing;
clock_t begin_reconstruction, end_reconstruction, clock_cycles_reconstruction;
clock_t begin_program, end_program, clock_cycles_program;

double execution_time_endpoints, execution_time_init_image, execution_time_DROP_iteration, execution_time_DROP, execution_time_update_calcs, execution_time_update_image, execution_time_tables;
double execution_time_preprocessing, execution_time_reconstruction, execution_time_program; 
std::vector<double> execution_times_DROP_iterations;
const char* execution_times_filename = "execution_times.txt";
FILE* execution_times_file;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------- Testing parameters/options controls ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
enum TX_OPTIONS{ FULL_TX, PARTIAL_TX, PARTIAL_TX_PREALLOCATED};
enum ENDPOINTS_ALGORITHMS{BOOL, NO_BOOL};
enum MLP_ALGORITHMS{STANDARD, TABULATED};
TX_OPTIONS ENDPOINTS_TX_MODE = FULL_TX;
TX_OPTIONS DROP_TX_MODE = FULL_TX;
ENDPOINTS_ALGORITHMS ENDPOINTS_ALG = NO_BOOL;
MLP_ALGORITHMS MLP_ALGORITHM = STANDARD;
//#define CONSTANT_LAMBDA_SCALE	VOXEL_WIDTH * LAMBDA
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------- Tabulated data file names ---------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char SIN_TABLE_FILENAME[]	= "sin_table.bin";							// Prefix of the file containing the tabulated values of sine function for angles [0, 2PI]
char COS_TABLE_FILENAME[]	= "cos_table.bin";							// Prefix of the file containing the tabulated values of cosine function for angles [0, 2PI]
char COEFFICIENT_FILENAME[]	= "coefficient.bin";						// Prefix of the file containing the tabulated values of the scattering coefficient for u_2-u_1/u_1 values in increments of 0.001
char POLY_1_2_FILENAME[]	= "poly_1_2.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {1,2,3,4,5,6} in increments of 0.001
char POLY_2_3_FILENAME[]	= "poly_2_3.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,3,4,5,6,7} in increments of 0.001
char POLY_3_4_FILENAME[]	= "poly_3_4.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,4,5,6,7,8} in increments of 0.001
char POLY_2_6_FILENAME[]	= "poly_2_6.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,6,12,20,30,42} in increments of 0.001
char POLY_3_12_FILENAME[]	= "poly_3_12.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,12,30,60,105,168} in increments of 0.001
#define TRIG_TABLE_MIN	-2 * PI											// [degrees] Minimum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_MAX	4 * PI											// [degrees] Maximum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_RANGE	(TRIG_TABLE_MAX - TRIG_TABLE_MIN)				// [degrees] Range of angles contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_STEP		(0.001 * ANGLE_TO_RADIANS)						// [degrees] Step of 1/4 degree for elements of sin/cos lookup table used for MLP
#define TRIG_TABLE_ELEMENTS	static_cast<int>(TRIG_TABLE_RANGE / TRIG_TABLE_STEP + 0.5)			// [#] # of elements contained in the sin/cos lookup table used for MLP
#define DEPTH_TABLE_RANGE	40.0										// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define DEPTH_TABLE_STEP		0.00005										// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define DEPTH_TABLE_ELEMENTS	static_cast<int>(DEPTH_TABLE_RANGE / DEPTH_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define DEPTH_TABLE_SHIFT	static_cast<int>(DEPTH_TABLE_RANGE / DEPTH_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP

#define POLY_TABLE_RANGE	40.0										// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_STEP		0.00005										// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_ELEMENTS	static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define POLY_TABLE_SHIFT	static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define INDEX_SHIFT_4_U_1	50											// [#] Difference between the index of consecutive elements of the poly table for u_1 dependent terms
//#define INDEX_SHIFT_4_U_1	0.05 / POLY_TABLE_STEP						// [#] Difference between the index of consecutive elements of the poly table for u_1 dependent terms

/***************************************************************************************************************************************************************************/
/*********************************************************** Memory allocation size for arrays (binning, image) ************************************************************/
/***************************************************************************************************************************************************************************/
#define SIZE_BINS_CHAR			( NUM_BINS   * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_BINS_BOOL			( NUM_BINS   * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_BINS_INT			( NUM_BINS   * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_UINT			( NUM_BINS   * sizeof(unsigned int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_FLOAT			( NUM_BINS	 * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_CHAR			( NUM_VOXELS * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_IMAGE_BOOL			( NUM_VOXELS * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_IMAGE_INT			( NUM_VOXELS * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_UINT			( NUM_VOXELS * sizeof(unsigned int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_FLOAT		( NUM_VOXELS * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_DOUBLE		( NUM_VOXELS * sizeof(double) )			// Amount of memory required for a floating point array used for binning
/***************************************************************************************************************************************************************************/
/************************************************************************* Precalculated Constants *************************************************************************/
/***************************************************************************************************************************************************************************/
#define PHI						((1 + sqrt(5.0) ) / 2)					// Positive golden ratio, positive solution of PHI^2-PHI-1 = 0; also PHI = a/b when a/b = (a + b) / a 
#define PHI_NEGATIVE			((1 - sqrt(5.0) ) / 2)					// Negative golden ratio, negative solution of PHI^2-PHI-1 = 0; 
#define PI_OVER_4				( atan( 1.0 ) )							// 1*pi/4 radians =   pi/4 radians = 45 degrees
#define PI_OVER_2				( 2 * atan( 1.0 ) )						// 2*pi/4 radians =   pi/2 radians = 90 degrees
#define THREE_PI_OVER_4			( 3 * atan( 1.0 ) )						// 3*pi/4 radians = 3*pi/4 radians = 135 degrees
#define PI						( 4 * atan( 1.0 ) )						// 4*pi/4 radians =   pi   radians = 180 degrees
#define FIVE_PI_OVER_4			( 5 * atan( 1.0 ) )						// 5*pi/4 radians = 5*pi/4 radians = 225 degrees
#define SIX_PI_OVER_4			( 5 * atan( 1.0 ) )						// 6*pi/4 radians = 3*pi/2 radians = 270 degrees
#define SEVEN_PI_OVER_4			( 7 * atan( 1.0 ) )						// 7*pi/4 radians = 7*pi/4 radians = 315 degrees
#define TWO_PI					( 8 * atan( 1.0 ) )						// 8*pi/4 radians = 2*pi   radians = 360 degrees = 0 degrees
#define ANGLE_TO_RADIANS		( PI/180.0 )							// Convertion from angle to radians
#define RADIANS_TO_ANGLE		( 180.0/PI )							// Convertion from radians to angle
#define ROOT_TWO				sqrtf(2.0)								// 2^(1/2) = Square root of 2 
#define MM_TO_CM				0.1										// 10 [mm] = 1 [cm] => 1 [mm] = 0.1 [cm]
#define VOXEL_ALLOWANCE			1.0e-7
#define START					true									// Used as an alias for true for starting timer
#define STOP					false									// Used as an alias for false for stopping timer
#define RIGHT					1										// Specifies that moving right corresponds with an increase in x position, used in voxel walk 
#define LEFT					-1										// Specifies that moving left corresponds with a decrease in x position, used in voxel walk 
#define UP						1										// Specifies that moving up corresponds with an increase in y/z position, used in voxel walk 
#define DOWN					-1										// Specifies that moving down corresponds with a decrease in y/z position, used in voxel walk 
#define BOOL_FORMAT				"%d"									// Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT				"%c"									// Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT				"%d"									// Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT			"%f"									// Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT			"%s"									// Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/********************************************************************* Preprocessing Array Declerations ********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************** Declaration of arrays number of histories per file, projection, angle, total, and translation **************************************/
/***************************************************************************************************************************************************************************/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;
//unsigned int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
//unsigned int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
//unsigned int* recon_vol_histories_per_projection;
//unsigned int histories_per_scan[NUM_SCANS];
//unsigned int post_cut_histories = 0;
/***************************************************************************************************************************************************************************/
/********************************************** Declaration of array used to store tracking plane distances from rotation axis *********************************************/
/***************************************************************************************************************************************************************************/
std::vector<double> projection_angles;
float SSD_u_Positions[8];
float* ut_entry_angle, * uv_entry_angle, * ut_exit_angle, * uv_exit_angle; 
int zero_WEPL = 0;
int zero_WEPL_files = 0;
/***************************************************************************************************************************************************************************/
/************************************************* Declaration of arrays for storage of input data for use on the host (_h) ************************************************/
/***************************************************************************************************************************************************************************/
int* gantry_angle_h, * bin_num_h, * bin_counts_h;
bool* missed_recon_volume_h, * failed_cuts_h;
float* t_in_1_h, * t_in_2_h, * t_out_1_h, * t_out_2_h;
float* u_in_1_h, * u_in_2_h, * u_out_1_h, * u_out_2_h;
float* v_in_1_h, * v_in_2_h, * v_out_1_h, * v_out_2_h;
float* ut_entry_angle_h, * ut_exit_angle_h;
float* uv_entry_angle_h, * uv_exit_angle_h;
float* x_entry_h, * y_entry_h, * z_entry_h;
float* x_exit_h, * y_exit_h, * z_exit_h;
float* xy_entry_angle_h, * xy_exit_angle_h;
float* xz_entry_angle_h, * xz_exit_angle_h;
float* relative_ut_angle_h, * relative_uv_angle_h;
//bool* entered_object_h, * exited_object_h;
float* WEPL_h;
/***************************************************************************************************************************************************************************/
/*********************************************** Declaration of arrays for storage of input data for use on the device (_d) ************************************************/
/***************************************************************************************************************************************************************************/
int* gantry_angle_d, * bin_num_d, * bin_counts_d;
bool* missed_recon_volume_d, * failed_cuts_d;
float* t_in_1_d, * t_in_2_d, * t_out_1_d, * t_out_2_d;
float* u_in_1_d, * u_in_2_d, * u_out_1_d, * u_out_2_d;
float* v_in_1_d, * v_in_2_d, * v_out_1_d, * v_out_2_d;
float* ut_entry_angle_d, * ut_exit_angle_d;
float* uv_entry_angle_d, * uv_exit_angle_d;
float* x_entry_d, * y_entry_d, * z_entry_d;
float* x_exit_d, * y_exit_d, * z_exit_d;
float* xy_entry_angle_d, * xy_exit_angle_d;
float* xz_entry_angle_d, * xz_exit_angle_d;
float* relative_ut_angle_d, * relative_uv_angle_d;
unsigned int* first_MLP_voxel_d;
//bool* entered_object_d, * exited_object_d;
float* WEPL_d;
int* voxel_x_d, voxel_y_d, voxel_z_d;
/***************************************************************************************************************************************************************************/
/********************************************** Declaration of statistical analysis arrays for use on host(_h) or device (_d) **********************************************/
/***************************************************************************************************************************************************************************/
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_energy_h, * mean_energy_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* mean_total_rel_angle_h, * mean_total_rel_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;
/***************************************************************************************************************************************************************************/
/******************************************** Declaration of pre/post filter sinogram for FBP for use on host(_h) or device (_d) *******************************************/
/***************************************************************************************************************************************************************************/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;
/***************************************************************************************************************************************************************************/
/***************************************************** Declaration of image arrays for use on host(_h) or device (_d) ******************************************************/
/***************************************************************************************************************************************************************************/
bool* SC_hull_h, * SC_hull_d;
bool* MSC_hull_h, * MSC_hull_d;
bool* SM_hull_h, * SM_hull_d;
bool* FBP_hull_h, * FBP_hull_d;
bool* hull_h, * hull_d;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* FBP_image_h, * FBP_image_d;
float* FBP_image_filtered_h, * FBP_image_filtered_d;
float* FBP_median_filtered_h, * FBP_median_filtered_d;
double* x_update_h;
float* x_update_d;
bool* intersected_hull_h, * intersected_hull_d;
unsigned int* num_voxel_intersections_h, * num_voxel_intersections_d;
unsigned int* S_h, * S_d;
unsigned int* block_voxels_h, *block_voxels_d;
unsigned int* block_counts_h, * block_counts_d;
float* x_h, * x_d;
unsigned int* global_a_i;
ULL* history_sequence;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------- Declaration of image arrays for use on host(_h) or device (_d) -----------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
FILE* sin_table_file, * cos_table_file, * scattering_table_file, * poly_1_2_file, * poly_2_3_file, * poly_3_4_file, * poly_2_6_file, * poly_3_12_file;
double* sin_table_h, * cos_table_h, * scattering_table_h, * poly_1_2_h, * poly_2_3_h, * poly_3_4_h, * poly_2_6_h, * poly_3_12_h;
double* sin_table_d, * cos_table_d, * scattering_table_d, * poly_1_2_d, * poly_2_3_d, * poly_3_4_d, * poly_2_6_d, * poly_3_12_d;
/***************************************************************************************************************************************************************************/
/********************************** Declaration of vectors used to accumulate data from histories that have passed currently applied cuts **********************************/
/***************************************************************************************************************************************************************************/
std::vector<int>	bin_num_vector;			
std::vector<int>	gantry_angle_vector;	
std::vector<float>	WEPL_vector;		
std::vector<float>	x_entry_vector;		
std::vector<float>	y_entry_vector;		
std::vector<float>	z_entry_vector;		
std::vector<float>	x_exit_vector;			
std::vector<float>	y_exit_vector;			
std::vector<float>	z_exit_vector;			
std::vector<float>	xy_entry_angle_vector;	
std::vector<float>	xz_entry_angle_vector;	
std::vector<float>	xy_exit_angle_vector;	
std::vector<float>	xz_exit_angle_vector;	
std::vector<unsigned int> first_MLP_voxel_vector;
//std::vector<float>	relative_ut_angle_vector;	
//std::vector<float>	relative_uv_angle_vector;
std::vector<int> voxel_x_vector;
std::vector<int> voxel_y_vector;
std::vector<int> voxel_z_vector;
/***************************************************************************************************************************************************************************/
/********************************** Declaration of arrays used to accumulate data from histories that have passed currently applied cuts ***********************************/
/***************************************************************************************************************************************************************************/
int*	bin_index;
float*	bin_WEPL;
int*	bin_num;			
int*	gantry_angle;	
float*	WEPL;		
float*	x_entry;		
float*	y_entry;		
float*	z_entry;		
float*	x_exit;			
float*	y_exit;			
float*	z_exit;			
float*	xy_entry_angle;	
float*	xz_entry_angle;	
float*	xy_exit_angle;	
float*	xz_exit_angle;	
float*	relative_ut_angle;	
float*	relative_uv_angle;
/***************************************************************************************************************************************************************************/
/*********************************************************************** Execution timer variables *************************************************************************/
/***************************************************************************************************************************************************************************/
clock_t program_start, program_end, pause_cycles = 0;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************* For Use In Development **************************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************************************************ MLP Test Image Parameters ************************************************************************/
/***************************************************************************************************************************************************************************/
int* test_image_h, test_image_d;


int MLP_IMAGE_COLUMNS = 100, MLP_IMAGE_ROWS = 100, MLP_IMAGE_SLICES = 5;
int MLP_IMAGE_VOXELS = MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS * MLP_IMAGE_SLICES;
int MLP_IMAGE_SIZE = MLP_IMAGE_VOXELS * sizeof(int);

int MLP_IMAGE_RECON_CYL_RADIUS_VOXELS = 50;
int MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS = 5;
int MLP_PHANTOM_A_VOXELS = 15, MLP_PHANTOM_B_VOXELS = 25;

double MLP_IMAGE_VOXEL_WIDTH = 0.1;
double MLP_IMAGE_VOXEL_HEIGHT = 0.1;
double MLP_IMAGE_VOXEL_THICKNESS = 1.0; 

double MLP_IMAGE_RECON_CYL_RADIUS = MLP_IMAGE_RECON_CYL_RADIUS_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_IMAGE_RECON_CYL_HEIGHT = MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS * MLP_IMAGE_VOXEL_THICKNESS;
double MLP_PHANTOM_A = MLP_PHANTOM_A_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_PHANTOM_B = MLP_PHANTOM_B_VOXELS * MLP_IMAGE_VOXEL_HEIGHT;

double MLP_IMAGE_WIDTH = MLP_IMAGE_COLUMNS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_IMAGE_HEIGHT = MLP_IMAGE_ROWS * MLP_IMAGE_VOXEL_HEIGHT;
double MLP_IMAGE_THICKNESS = MLP_IMAGE_SLICES * MLP_IMAGE_VOXEL_THICKNESS;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************ End of Parameter Definitions *********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
//#endif // _PCT_RECONSTRUCTION_H_
