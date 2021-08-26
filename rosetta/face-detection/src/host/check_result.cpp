/*===============================================================*/
/*                                                               */
/*                       check_result.cpp                        */
/*                                                               */
/*      Software evaluation of training and test error rate      */
/*                                                               */
/*===============================================================*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include "image.h"
#include "typedefs.h"

MyRect expected[] =
{
  {50, 89, 35, 35},
  {48, 91, 35, 35},
  {49, 91, 35, 35},
  {50, 91, 35, 35},
  {48, 92, 35, 35},
  {98, 94, 35, 35},
  {43, 86, 41, 41},
  {45, 86, 41, 41},
  {47, 86, 41, 41},
  {48, 86, 41, 41},
  {140, 86, 41, 41},
  {173, 86, 41, 41},
  {175, 86, 41, 41},
  {176, 86, 41, 41},
  {43, 88, 41, 41},
  {45, 88, 41, 41},
  {47, 88, 41, 41},
  {48, 88, 41, 41},
  {92, 88, 41, 41},
  {93, 88, 41, 41},
  {92, 90, 41, 41},
  {41, 81, 50, 50},
  {131, 81, 50, 50},
  {133, 81, 50, 50},
  {172, 81, 50, 50},
  {41, 83, 50, 50},
  {44, 83, 50, 50},
  {131, 83, 50, 50},
  {133, 83, 50, 50},
  {135, 83, 50, 50},
  {170, 83, 50, 50},
  {89, 85, 50, 50},
  {133, 85, 50, 50},
  {127, 75, 60, 60},
  {124, 77, 60, 60},
  {127, 77, 60, 60},
  {129, 77, 60, 60},
  {124, 80, 60, 60},
  {127, 80, 60, 60},
  {129, 80, 60, 60},
  {127, 82, 60, 60},
  {129, 82, 60, 60},
  {119, 72, 72, 72}
};

bool compare_rectangle(const MyRect & rect_1, const MyRect & rect_2)
{
  if (rect_1.width != rect_2.width)
    return rect_1.width < rect_2.width;
  else if (rect_1.y != rect_2.y)
    return rect_1.y < rect_2.y;
  else
    return rect_1.x < rect_2.x;
}

bool check_results(int &result_size, 
                   int result_x[RESULT_SIZE], 
                   int result_y[RESULT_SIZE], 
                   int result_w[RESULT_SIZE],  
                   int result_h[RESULT_SIZE],
                   unsigned char Data[IMAGE_HEIGHT][IMAGE_WIDTH],
                   std::string outFile)
{
  printf("\nresult_size = %d", result_size);

  MyRect result[RESULT_SIZE];

  for (int j = 0; j < RESULT_SIZE; j++){
    result[j].x = result_x[j];
    result[j].y = result_y[j];
    result[j].width = result_w[j];
    result[j].height = result_h[j];
  }

  std::sort(result, result + result_size, compare_rectangle);

  bool error = result_size != sizeof(expected) / sizeof(expected[0]);
  for( int i=0 ; i < result_size ; i++ )
  {
    printf("\n [Test Bench (main) ] detected rects: %d %d %d %d",
           result[i].x,result[i].y,result[i].width,result[i].height);
    if (!error && (result[i].x != expected[i].x ||
                   result[i].y != expected[i].y ||
                   result[i].width != expected[i].width ||
                   result[i].height != expected[i].height))
      error = true;
  }
 
  printf("\n-- saving output image [Start] --\r\n"); 

  // Draw the rectangles onto the images and save the outputs.
  for(int i = 0; i < result_size ; i++ )
  {
    MyRect r = result[i];
    drawRectangle(Data, r);
  }

  int flag = writePgm(outFile.c_str(), Data); 

  printf("\n-- saving output image [Done] --\r\n");

  return error;
}
