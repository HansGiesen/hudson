/*===============================================================*/
/*                                                               */
/*                        check_result.h                         */
/*                                                               */
/*      Compare result and expected label to get error rate      */
/*                                                               */
/*===============================================================*/

#ifndef SW
bool check_results(axi_bus* output);
#else
bool check_results(bit8 output[MAX_X][MAX_Y]);
#endif
