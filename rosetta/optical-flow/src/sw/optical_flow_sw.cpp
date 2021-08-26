/*===============================================================*/
/*                                                               */
/*                     optical_flow_sw.cpp                       */
/*                                                               */
/*              Software version for optical flow.               */
/*                                                               */
/*===============================================================*/

#include "optical_flow_sw.h"
#include <cstdio>
#include <iostream>

//#define OUTPUT_FRAMES
//#define OUTPUT_RANGES

template<typename type> void update_range(type value, double & min,
                                          double & max)
{
  if (value > max)
    max = value;
  if (value < min)
    min = value;
}

template<> void update_range(gradient_t value, double & min, double & max)
{
  update_range(value.x, min, max);
  update_range(value.y, min, max);
  update_range(value.z, min, max);
}

template<> void update_range(outer_t value, double & min, double & max)
{
  for (int i = 0; i < 6; i++)
    update_range(value.val[i], min, max);
}

template<> void update_range(tensor_t value, double & min, double & max)
{
  for (int i = 0; i < 6; i++)
    update_range(value.val[i], min, max);
}

template<> void update_range(velocity_t value, double & min, double & max)
{
  update_range(value.x, min, max);
  update_range(value.y, min, max);
}

template<typename type> void print_range(const char * name,
                                         type frame[MAX_HEIGHT][MAX_WIDTH])
{
#ifdef OUTPUT_RANGES
  double min = 0.0;
  double max = 0.0;
  std::cout << name << '\n';
  for (int y = 0; y < MAX_HEIGHT; y++)
    for (int x = 0; x < MAX_WIDTH; x++)
      update_range(frame[y][x], min, max);
  std::cout << "Minimum: " << min << " Maximum: " << max << '\n';
#endif
}

template<typename type> void print_value(type value)
{
  std::cout << value;
}

template<> void print_value(gradient_t value)
{
  std::cout << '(' << value.x << ", " << value.y << ", " << value.z << ")";
}

template<> void print_value(outer_t value)
{
  std::cout << '(';
  for (int i = 0; i < 6; i++)
  {
    std::cout << value.val[i];
    if (i != 5)
      std::cout << ", ";
  }
  std::cout << ")";
}

template<> void print_value(tensor_t value)
{
  std::cout << '(';
  for (int i = 0; i < 6; i++)
  {
    std::cout << value.val[i];
    if (i != 5)
      std::cout << ", ";
  }
  std::cout << ")";
}

template<> void print_value(velocity_t value)
{
  std::cout << '(' << value.x << ", " << value.y << ")";
}

template<typename type> void print_frame(const char * name,
                                         type frame[MAX_HEIGHT][MAX_WIDTH])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    {
      print_value(frame[y][x]);
      if (x != MAX_WIDTH - 1)
        std::cout << ", ";
    }
    std::cout << '\n';
  }
#endif
  print_range(name, frame);
}

// compute x, y gradient
void gradient_xy_calc(pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH])
{
  pixel_t x_grad, y_grad;
  for (int r = 0; r < MAX_HEIGHT + 2; r ++ )
  {
    for (int c = 0; c < MAX_WIDTH + 2; c ++)
    {
      x_grad = 0;
      y_grad = 0;
      if (r >= 4 && r < MAX_HEIGHT && c >= 4 && c < MAX_WIDTH)
      {
        for (int i = 0; i < 5; i++)
        {
          x_grad += frame[r-2][c-i] * GRAD_WEIGHTS[4-i];
          y_grad += frame[r-i][c-2] * GRAD_WEIGHTS[4-i];
        }
        gradient_x[r-2][c-2] = x_grad / 12;
        gradient_y[r-2][c-2] = y_grad / 12;
      }
      else if (r >= 2 && c >= 2)
      {
        gradient_x[r-2][c-2] = 0;
        gradient_y[r-2][c-2] = 0;
      }
    }
  }
}

// compute z gradient
void gradient_z_calc(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH], 
                     pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH; c ++)
    {
      gradient_z[r][c] = 0.0f;
      gradient_z[r][c] += frame0[r][c] * GRAD_WEIGHTS[0]; 
      gradient_z[r][c] += frame1[r][c] * GRAD_WEIGHTS[1]; 
      gradient_z[r][c] += frame2[r][c] * GRAD_WEIGHTS[2]; 
      gradient_z[r][c] += frame3[r][c] * GRAD_WEIGHTS[3]; 
      gradient_z[r][c] += frame4[r][c] * GRAD_WEIGHTS[4]; 
      gradient_z[r][c] /= 12.0f;
    }
  }
}

// compute y weight
void gradient_weight_y(pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT + 3; r ++)
  {
    for (int c = 0; c < MAX_WIDTH; c ++)
    {
      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if (r >= 6 && r < MAX_HEIGHT)
      { 
        for (int i = 0; i < 7; i ++)
        {
          acc.x += gradient_x[r-i][c] * GRAD_FILTER[i];
          acc.y += gradient_y[r-i][c] * GRAD_FILTER[i];
          acc.z += gradient_z[r-i][c] * GRAD_FILTER[i];
        }
        filt_grad[r-3][c] = acc;            
      }
      else if (r >= 3)
      {
        filt_grad[r-3][c] = acc;
      }
    }
  }
}

// compute x weight
void gradient_weight_x(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH + 3; c ++)
    {
      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if (c >= 6 && c < MAX_WIDTH)
      {
        for (int i = 0; i < 7; i ++)
        {
          acc.x += y_filt[r][c-i].x * GRAD_FILTER[i];
          acc.y += y_filt[r][c-i].y * GRAD_FILTER[i];
          acc.z += y_filt[r][c-i].z * GRAD_FILTER[i];
        }
        filt_grad[r][c-3] = acc;
      }
      else if (c >= 3)
      {
        filt_grad[r][c-3] = acc;
      }
    }
  }
}
 
// outer product
void outer_product(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
                   outer_t outer_product[MAX_HEIGHT][MAX_WIDTH])
{ 
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH; c ++)
    {
      gradient_t grad = gradient[r][c];
      outer_t out;
      out.val[0] = grad.x * grad.x;
      out.val[1] = grad.y * grad.y;
      out.val[2] = grad.z * grad.z;
      out.val[3] = grad.x * grad.y;
      out.val[4] = grad.x * grad.z;
      out.val[5] = grad.y * grad.z;
      outer_product[r][c] = out;
    }
  }
}

// tensor weight y
void tensor_weight_y(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT + 1; r ++)
  {
    for(int c = 0; c < MAX_WIDTH; c ++)
    {
      tensor_t acc;
      for (int k = 0; k < 6; k ++)
      {
        acc.val[k] = 0;
      }

      if (r >= 2 && r < MAX_HEIGHT) 
      {
        for (int i = 0; i < 3; i ++)
        {
          for(int component = 0; component < 6; component ++)
          {
            acc.val[component] += outer[r-i][c].val[component] * TENSOR_FILTER[i];
          }
        }
      }
      if (r >= 1)
      { 
        tensor_y[r-1][c] = acc;            
      }
    }
  }
}

// tensor weight x
void tensor_weight_x(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    for (int c = 0; c < MAX_WIDTH + 1; c ++)
    {
      tensor_t acc;
      for(int k = 0; k < 6; k++)
      {
        acc.val[k] = 0;
      }
      if (c >= 2 && c < MAX_WIDTH) 
      {
        for (int i = 0; i < 3; i ++)
        {
          for (int component = 0; component < 6; component ++)
          {
            acc.val[component] += tensor_y[r][c-i].val[component] * TENSOR_FILTER[i];
          }
        }
      }
      if (c >= 1)
      {
        tensor[r][c-1] = acc;
      }
    }
  }
}

// compute flow
void flow_calc(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],
               velocity_t output[MAX_HEIGHT][MAX_WIDTH])
{
  for(int r = 0; r < MAX_HEIGHT; r ++)
  {
    for(int c = 0; c < MAX_WIDTH; c ++)
    {
      if (r >= 2 && r < MAX_HEIGHT - 2 && c >= 2 && c < MAX_WIDTH - 2)
      {
        pixel_t denom = tensors[r][c].val[0] * tensors[r][c].val[1] -
                        tensors[r][c].val[3] * tensors[r][c].val[3];
        output[r][c].x = (tensors[r][c].val[5] * tensors[r][c].val[3] -
                          tensors[r][c].val[4] * tensors[r][c].val[1]) / denom;
        output[r][c].y = (tensors[r][c].val[4] * tensors[r][c].val[3] -
                          tensors[r][c].val[5] * tensors[r][c].val[0]) / denom;
      }
      else
      {
        output[r][c].x = 0;
        output[r][c].y = 0;
      }
    }
  }
}

// top-level sw function
void optical_flow_sw(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
                     velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  // intermediate arrays
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];

  // compute
  print_frame("frame2", frame2);
  gradient_xy_calc(frame2, gradient_x, gradient_y);
  print_frame("gradient_x", gradient_x);
  print_frame("gradient_y", gradient_y);
  gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z);
  print_frame("gradient_z", gradient_z);
  gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
  print_frame("y_filtered", y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  print_frame("filtered_gradient", filtered_gradient);
  outer_product(filtered_gradient, out_product);
  print_frame("out_product", out_product);
  tensor_weight_y(out_product, tensor_y);
  print_frame("tensor_y", tensor_y);
  tensor_weight_x(tensor_y, tensor);
  print_frame("tensor", tensor);
  flow_calc(tensor, outputs);
  print_frame("outputs", outputs);
}

