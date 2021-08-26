/*===============================================================*/
/*                                                               */
/*                      optical_flow.cpp                         */
/*                                                               */
/*             Hardware function for optical flow                */
/*                                                               */
/*===============================================================*/

#include "optical_flow.h"
#include <iostream>

// use HLS fixed point
#include "ap_fixed.h"

//#define OUTPUT_FRAMES

const int offs_x = MAX_WIDTH - 20;
const int offs_y = MAX_HEIGHT - 20;

template <int width> void print_frame(const char * name,
    ap_uint<PAR_FACTOR * width> frame[MAX_HEIGHT][MAX_WIDTH / PAR_FACTOR])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    { 
      ap_uint<PAR_FACTOR * width> vec = frame[offs_y + y][(offs_x + x) / PAR_FACTOR];
      vec >>= (((offs_x + x) % PAR_FACTOR) * width);
      ap_uint<width> value;
      value(width - 1, 0) = vec(width - 1, 0);
      std::cout << value.to_double();
      if (x != 19)
        std::cout << ", ";   
    }
    std::cout << '\n';
  }
#endif  
}

void print_frame(const char * name,
                 gradient_vec_t frame[MAX_HEIGHT][MAX_WIDTH / PAR_FACTOR])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    {
      gradient_vec_t vec = frame[offs_y + y][(offs_x + x) / PAR_FACTOR];
      vec.x >>= (((offs_x + x) % PAR_FACTOR) * pixel_t::width);
      vec.y >>= (((offs_x + x) % PAR_FACTOR) * pixel_t::width);
      vec.z >>= (((offs_x + x) % PAR_FACTOR) * pixel_t::width);
      gradient_t grad;
      grad.x(pixel_t::width - 1, 0) = vec.x(pixel_t::width - 1, 0);
      grad.y(pixel_t::width - 1, 0) = vec.y(pixel_t::width - 1, 0);
      grad.z(pixel_t::width - 1, 0) = vec.z(pixel_t::width - 1, 0);
      std::cout << '(' << grad.x.to_double() << ", "
                       << grad.y.to_double() << ", "
                       << grad.z.to_double() << ')';
      if (x != 19)
        std::cout << ", ";
    }
    std::cout << '\n';
  }
#endif
}

void print_frame(const char * name,
                 outer_vec_t frame[MAX_HEIGHT][MAX_WIDTH / PAR_FACTOR])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    {
      outer_vec_t vec = frame[offs_y + y][(offs_x + x) / PAR_FACTOR];
      std::cout << '(';
      for (int i = 0; i < 6; i++)
      {
        vec.val[i] >>= (((offs_x + x) % PAR_FACTOR) * outer_pixel_t::width);
        outer_pixel_t outer;
        outer(outer_pixel_t::width - 1, 0) =
            vec.val[i](outer_pixel_t::width - 1, 0);
        std::cout << outer.to_double();
        if (i < 5)
          std::cout << ", ";
      }
      std::cout << ')';
      if (x != 19)
        std::cout << ", ";
    }
    std::cout << '\n';
  }
#endif
}

void print_frame(const char * name,
                 tensor_vec_t frame[MAX_HEIGHT][MAX_WIDTH / PAR_FACTOR])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    {
      tensor_vec_t vec = frame[offs_y + y][(offs_x + x) / PAR_FACTOR];
      std::cout << '(';
      for (int i = 0; i < 6; i++)
      {
        vec.val[i] >>= (((offs_x + x) % PAR_FACTOR) * tensor_pixel_t::width);
        tensor_pixel_t tensor;
        tensor(tensor_pixel_t::width - 1, 0) =
            vec.val[i](tensor_pixel_t::width - 1, 0);
        std::cout << tensor.to_double();
        if (i < 5)
          std::cout << ", ";
      }
      std::cout << ')';
      if (x != 19)
        std::cout << ", ";
    }
    std::cout << '\n';
  }
#endif
}

void print_frame(const char * name,
                 velocity_vec_t frame[MAX_HEIGHT][MAX_WIDTH / PAR_FACTOR])
{
#ifdef OUTPUT_FRAMES
  std::cout << name << '\n';
  for (int y = 0; y < 20; y++)
  {
    for (int x = 0; x < 20; x++)
    {
      velocity_vec_t vec = frame[offs_y + y][(offs_x + x) / PAR_FACTOR];
      vec.x >>= (((offs_x + x) % PAR_FACTOR) * vel_pixel_t::width);
      vec.y >>= (((offs_x + x) % PAR_FACTOR) * vel_pixel_t::width);
      velocity_t vel;
      vel.x(vel_pixel_t::width - 1, 0) = vec.x(vel_pixel_t::width - 1, 0);
      vel.y(vel_pixel_t::width - 1, 0) = vec.y(vel_pixel_t::width - 1, 0);
      std::cout << '(' << vel.x.to_double() << ", "
                       << vel.y.to_double() << ')';
      if (x != 19)
        std::cout << ", ";
    }
    std::cout << '\n';
  }
#endif
}

// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH/PAR_FACTOR; 
const int default_depth = MAX_WIDTH/PAR_FACTOR;

void stream_in(axi_bus_t input[INPUT_LENGTH],
               input_vec_t frame1_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               input_vec_t frame2_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               input_vec_t frame3_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               input_vec_t frame3_b[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               input_vec_t frame4_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               input_vec_t frame5_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  for (int r=0; r<MAX_HEIGHT; r++)
    for (int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1
      axi_bus_t input_word = input[r * MAX_WIDTH/PAR_FACTOR + c];
  
      input_vec_t output_words[5];
      for (int p=0; p<PAR_FACTOR; p++)
      {
        axi_bus_t word = input_word; 
        for (int frame=0; frame<5; frame++)
        {
          output_words[frame] >>= 8;
          output_words[frame](8*PAR_FACTOR-1, 8*PAR_FACTOR-8) = word(7, 0);
          word >>= 8;
        }
        input_word >>= 64;
      }
      
      frame1_a[r][c] = output_words[0];
      frame2_a[r][c] = output_words[1];
      frame3_a[r][c] = output_words[2];
      frame3_b[r][c] = output_words[2];
      frame4_a[r][c] = output_words[3];
      frame5_a[r][c] = output_words[4];
    }
}

// calculate gradient in x and y directions
void gradient_xy_calc(input_vec_t frame[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    pixel_vec_t gradient_x[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    pixel_vec_t gradient_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  // our own line buffer
  input_vec_t buf[5][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS array_partition variable=buf complete dim=1

  // window buffer
  input_t window[5][5*PAR_FACTOR];
  #pragma HLS array_partition variable=window complete dim=0

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  // HG: Vivado HLS 2019.1 creates an invalid implementation if the divisor is
  // an integer.
  const int GRAD_DIVISOR = 12;

  GRAD_XY_OUTER: for(int r=0; r<MAX_HEIGHT+2; r++)
  {
    GRAD_XY_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR+2; c++)
    {
      #pragma HLS pipeline II=1

      // update line buffer
      if (c < MAX_WIDTH/PAR_FACTOR)
        for (int i = 0; i < 4; i++)
          buf[i][c] = buf[i+1][c];

      // the new value is either 0 or read from frame
      if (r<MAX_HEIGHT && c<MAX_WIDTH/PAR_FACTOR)
        buf[4][c] = frame[r][c];
      else
        buf[4][c] = 0;

      // manage window buffer
      for (int i=0; i<5; i++)
      {
        for (int j=0; j<4*PAR_FACTOR; j++)
          window[i][j] = window[i][j+PAR_FACTOR];

        input_vec_t vec = buf[i][c];
        for (int j=0; j<PAR_FACTOR; j++)
        {
          window[i][4*PAR_FACTOR+j](7, 0) = vec(7, 0);
          vec >>= 8;
        }
      }

      // compute gradient
      pixel_t x_grad[PAR_FACTOR];
      pixel_t y_grad[PAR_FACTOR];
      for (int p=0; p<PAR_FACTOR; p++)
      {
        x_grad[p] = 0;
        y_grad[p] = 0;
      }

      for(int p=0; p<PAR_FACTOR; p++)
      {
        int cc = PAR_FACTOR * c + p;

        if(r>=4 && r<MAX_HEIGHT && cc>=2*PAR_FACTOR+2 && cc<MAX_WIDTH+2*PAR_FACTOR-2)
        {
          GRAD_XY_XYGRAD: for(int i=0; i<5; i++)
          {
            x_grad[p] += window[2][2*PAR_FACTOR+p-2+i]*GRAD_WEIGHTS[i];
            y_grad[p] += window[i][2*PAR_FACTOR+p]*GRAD_WEIGHTS[i];
          }
          x_grad[p] /= GRAD_DIVISOR;
          y_grad[p] /= GRAD_DIVISOR;
        }
        else if(r>=2 && c>=2)
        {
          x_grad[p] = 0;
          y_grad[p] = 0;
        }
      }

      pixel_vec_t x_grad_vec = 0;
      pixel_vec_t y_grad_vec = 0;
      for (int p=PAR_FACTOR-1; p>=0; p--)
      {
        x_grad_vec <<= PIX_WIDTH;
        y_grad_vec <<= PIX_WIDTH;
        x_grad_vec = x_grad_vec | x_grad[p](PIX_WIDTH-1, 0);
        y_grad_vec = y_grad_vec | y_grad[p](PIX_WIDTH-1, 0);
      }
      if (r>=2 && c>=2)
      {
        gradient_x[r-2][c-2] = x_grad_vec;
        gradient_y[r-2][c-2] = y_grad_vec;
      }
    }
  }
}

// calculate gradient in the z direction
void gradient_z_calc(input_vec_t frame1[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    input_vec_t frame2[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    input_vec_t frame3[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    input_vec_t frame4[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    input_vec_t frame5[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    pixel_vec_t gradient_z[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  // HG: Vivado HLS 2019.1 creates an invalid implementation if the divisor is
  // an unsigned integer.
  const int GRAD_DIVISOR = 12;
  GRAD_Z_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_Z_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1
      input_vec_t inputs[5];
      #pragma HLS array_partition variable=inputs complete dim=0
      inputs[0] = frame1[r][c];
      inputs[1] = frame2[r][c];
      inputs[2] = frame3[r][c];
      inputs[3] = frame4[r][c];
      inputs[4] = frame5[r][c];
      pixel_vec_t output;
      for(int p=0; p<PAR_FACTOR; p++)
      {
        input_t pixels[5];
        #pragma HLS array_partition variable=pixels complete dim=0
        for(int i=0; i<5; i++)
        {
          pixels[i](7,0) = inputs[i](7,0);
          inputs[i] >>= 8;
        }

        pixel_t result = ((pixel_t)(pixels[0]*GRAD_WEIGHTS[0]
                                  + pixels[1]*GRAD_WEIGHTS[1]
                                  + pixels[2]*GRAD_WEIGHTS[2]
                                  + pixels[3]*GRAD_WEIGHTS[3]
                                  + pixels[4]*GRAD_WEIGHTS[4]))/GRAD_DIVISOR;

        output >>= PIX_WIDTH;
        int msb = PAR_FACTOR * PIX_WIDTH - 1;
        output(msb,msb-PIX_WIDTH+1) = result(PIX_WIDTH-1,0);
      }
      gradient_z[r][c] = output;
    }
  }
}

// average the gradient in y direction
void gradient_weight_y(pixel_vec_t gradient_x[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    pixel_vec_t gradient_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    pixel_vec_t gradient_z[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
    gradient_vec_t filt_grad[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  gradient_vec_t buf[7][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS array_partition variable=buf complete dim=1

  const grad_filt_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+3; r++)
  {
    GRAD_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1

      for(int i=0; i<6; i++)
        buf[i][c] = buf[i+1][c];

      gradient_vec_t tmp;
      if(r<MAX_HEIGHT)
      {
        tmp.x = gradient_x[r][c];
        tmp.y = gradient_y[r][c];
        tmp.z = gradient_z[r][c];
      }
      else
      {
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
      }
      buf[6][c] = tmp;

      gradient_vec_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if(r >= 6 && r<MAX_HEIGHT)
      {
        gradient_t accs[PAR_FACTOR];
        for(int p=0; p<PAR_FACTOR; p++)
        {
          accs[p].x = 0;
          accs[p].y = 0;
          accs[p].z = 0;
        }

        GRAD_WEIGHT_Y_ACC: for(int i=0; i<7; i++)
        {
          gradient_vec_t grad_vec = buf[i][c];
          gradient_t grad;
          for(int p=0; p<PAR_FACTOR; p++)
          {
            grad.x(PIX_WIDTH-1, 0) = grad_vec.x(PIX_WIDTH-1, 0);
            grad.y(PIX_WIDTH-1, 0) = grad_vec.y(PIX_WIDTH-1, 0);
            grad.z(PIX_WIDTH-1, 0) = grad_vec.z(PIX_WIDTH-1, 0);
            accs[p].x += grad.x*GRAD_FILTER[i];
            accs[p].y += grad.y*GRAD_FILTER[i];
            accs[p].z += grad.z*GRAD_FILTER[i];
            grad_vec.x >>= PIX_WIDTH;
            grad_vec.y >>= PIX_WIDTH;
            grad_vec.z >>= PIX_WIDTH;
          }
        }
        for(int p=PAR_FACTOR-1; p>=0; p--)
        {
          acc.x <<= PIX_WIDTH;
          acc.y <<= PIX_WIDTH;
          acc.z <<= PIX_WIDTH;
          acc.x(PIX_WIDTH-1,0) = accs[p].x(PIX_WIDTH-1,0);
          acc.y(PIX_WIDTH-1,0) = accs[p].y(PIX_WIDTH-1,0);
          acc.z(PIX_WIDTH-1,0) = accs[p].z(PIX_WIDTH-1,0);
        }
        filt_grad[r-3][c] = acc;
      }
      else if(r>=3)
      {
        filt_grad[r-3][c] = acc;
      }
    }
  }
}

// average gradient in the x direction
void gradient_weight_x(gradient_vec_t y_filt[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
                       gradient_vec_t filt_grad[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  gradient_t buf[7*PAR_FACTOR];
  const grad_filt_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR+3; c++)
    {
      #pragma HLS pipeline II=1
      for (int i=0; i<6*PAR_FACTOR; i++)
        buf[i] = buf[i+PAR_FACTOR];
      gradient_vec_t tmp;
      if(c<MAX_WIDTH/PAR_FACTOR)
      {
        tmp = y_filt[r][c];
      }
      else
      {
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
      }
      for (int i=0; i<PAR_FACTOR; i++)
      {
        gradient_t grad;
        grad.x(PIX_WIDTH-1, 0) = tmp.x(PIX_WIDTH-1, 0);
        grad.y(PIX_WIDTH-1, 0) = tmp.y(PIX_WIDTH-1, 0);
        grad.z(PIX_WIDTH-1, 0) = tmp.z(PIX_WIDTH-1, 0);
        buf[i+6*PAR_FACTOR] = grad;
        tmp.x >>= PIX_WIDTH;
        tmp.y >>= PIX_WIDTH;
        tmp.z >>= PIX_WIDTH;
      }

      gradient_t accs[PAR_FACTOR];
      for (int p=0; p<PAR_FACTOR; p++)
      {
        accs[p].x = 0;
        accs[p].y = 0;
        accs[p].z = 0;
      }
      for (int p=0; p<PAR_FACTOR; p++)
      {
        int cc = PAR_FACTOR * c + p;
        if(cc>=3*PAR_FACTOR+3 && cc<MAX_WIDTH+3*PAR_FACTOR-3)
        {
          GRAD_WEIGHT_X_ACC: for(int i=0; i<7; i++)
          {
            accs[p].x += buf[i+p+3*PAR_FACTOR-3].x*GRAD_FILTER[i];
            accs[p].y += buf[i+p+3*PAR_FACTOR-3].y*GRAD_FILTER[i];
            accs[p].z += buf[i+p+3*PAR_FACTOR-3].z*GRAD_FILTER[i];
          }
        }
      }
      gradient_vec_t acc;
      for (int p=PAR_FACTOR-1; p>=0; p--)
      {
        acc.x <<= PIX_WIDTH;
        acc.y <<= PIX_WIDTH;
        acc.z <<= PIX_WIDTH;
        acc.x(PIX_WIDTH-1, 0) = accs[p].x(PIX_WIDTH-1, 0);
        acc.y(PIX_WIDTH-1, 0) = accs[p].y(PIX_WIDTH-1, 0);
        acc.z(PIX_WIDTH-1, 0) = accs[p].z(PIX_WIDTH-1, 0);
      }
      if (c>=3)
        filt_grad[r][c-3] = acc;
    }
  }
}

// outer product 
void outer_product(gradient_vec_t gradient[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
     outer_vec_t outer_product[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  OUTER_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    OUTER_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1
      gradient_vec_t grad_vec = gradient[r][c];
      outer_vec_t out_vec;
      for(int p=0; p<PAR_FACTOR; p++)
      {
        gradient_t grad;
        grad.x(PIX_WIDTH-1, 0) = grad_vec.x(PIX_WIDTH-1, 0);
        grad.y(PIX_WIDTH-1, 0) = grad_vec.y(PIX_WIDTH-1, 0);
        grad.z(PIX_WIDTH-1, 0) = grad_vec.z(PIX_WIDTH-1, 0);
        grad_vec.x >>= PIX_WIDTH;
        grad_vec.y >>= PIX_WIDTH;
        grad_vec.z >>= PIX_WIDTH;

        outer_t out;
        out.val[0] = (grad.x*grad.x);
        out.val[1] = (grad.y*grad.y);
        out.val[2] = (grad.z*grad.z);
        out.val[3] = (grad.x*grad.y);
        out.val[4] = (grad.x*grad.z);
        out.val[5] = (grad.y*grad.z);

        for(int i=0; i<6; i++)
        {
          out_vec.val[i] >>= OUTER_WIDTH;
          int msb = OUTER_WIDTH*PAR_FACTOR-1;
          out_vec.val[i](msb,msb-OUTER_WIDTH+1) = out.val[i](OUTER_WIDTH-1, 0);
        }
      }
      outer_product[r][c] = out_vec;
    }
  }
}

// tensor weight
void tensor_weight_y(outer_vec_t outer[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
                     tensor_vec_t tensor_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  #pragma HLS data_pack variable=outer
  #pragma HLS data_pack variable=tensor_y
  outer_vec_t buf[3][MAX_WIDTH/PAR_FACTOR];
  const tensor_filt_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+1; r++)
  {
    TENSOR_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1

      outer_vec_t tmp_vec;
      for(int i=0; i<2; i++)
        buf[i][c] = buf[i+1][c];
      if(r<MAX_HEIGHT)
      {
        tmp_vec = outer[r][c];
      }
      else
      {
        TENSOR_WEIGHT_Y_TMP_INIT: for(int i=0; i<6; i++)
          tmp_vec.val[i] = 0;
      }
      buf[2][c] = tmp_vec;

      tensor_vec_t acc;
      TENSOR_WEIGHT_Y_ACC_INIT: for(int k =0; k<6; k++)
        acc.val[k] = 0;

      if (r >= 2 && r < MAX_HEIGHT)
      {
        tensor_t accs[PAR_FACTOR];
        for (int p=0; p<PAR_FACTOR; p++)
          for(int i=0; i<6; i++)
            accs[p].val[i] = 0;

        TENSOR_WEIGHT_Y_TMP_OUTER: for(int i=0; i<3; i++)
        {
          tmp_vec = buf[i][c];
          for (int p=0; p<PAR_FACTOR; p++)
          {
            TENSOR_WEIGHT_Y_TMP_INNER: for(int component=0; component<6; component++)
            {
              outer_pixel_t tmp;
              tmp(OUTER_WIDTH-1, 0) = tmp_vec.val[component](OUTER_WIDTH-1, 0);
              accs[p].val[component] += tmp*TENSOR_FILTER[i];
              tmp_vec.val[component] >>= OUTER_WIDTH;
            }
          }
        }
        for (int p=PAR_FACTOR-1; p>=0; p--)
        {
          for(int i=0; i<6; i++)
          {
            acc.val[i] <<= TENSOR_WIDTH;
            acc.val[i](TENSOR_WIDTH-1, 0) = accs[p].val[i](TENSOR_WIDTH-1, 0);
          }
        }
      }
      if(r >= 1)
      {
        tensor_y[r-1][c] = acc;
      }
    }
  }
}

void tensor_weight_x(tensor_vec_t tensor_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
                     tensor_vec_t tensor[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  #pragma HLS data_pack variable=tensor_y
  #pragma HLS data_pack variable=tensor
  tensor_t buf[3*PAR_FACTOR];
  const tensor_filt_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    TENSOR_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR+1; c++)
    {
      #pragma HLS pipeline II=1
      for(int i=0; i<2*PAR_FACTOR; i++)
        buf[i] = buf[i + PAR_FACTOR];

      tensor_vec_t tmp_vec;
      if(c<MAX_WIDTH/PAR_FACTOR)
      {
        tmp_vec = tensor_y[r][c];
      }
      else
      {
        TENSOR_WEIGHT_X_TMP_INIT: for(int i=0; i<6; i++)
          tmp_vec.val[i] = 0;
      }

      for(int i=0; i<PAR_FACTOR; i++)
      {
        for(int k =0; k<6; k++)
        {
          buf[i + 2 * PAR_FACTOR].val[k](TENSOR_WIDTH - 1, 0) = tmp_vec.val[k](TENSOR_WIDTH - 1, 0);
          tmp_vec.val[k] >>= TENSOR_WIDTH;
        }
      }

      tensor_vec_t acc_vec;
      for(int p=PAR_FACTOR-1; p>=0; p--)
      {
        tensor_t acc;
        TENSOR_WEIGHT_X_ACC_INIT: for(int k =0; k<6; k++)
          acc.val[k] = 0;

        int cc = PAR_FACTOR * c + p;
        if (cc >= PAR_FACTOR+1 && cc < MAX_WIDTH+PAR_FACTOR-1)
        {
          TENSOR_WEIGHT_X_TMP_OUTER: for(int i=0; i<3; i++)
          {
            tensor_t tmp = buf[i+p+PAR_FACTOR-1];
            TENSOR_WEIGHT_X_TMP_INNER: for(int component=0; component<6; component++)
            {
              acc.val[component] += tmp.val[component]*TENSOR_FILTER[i];
            }
          }
        }

        for(int k=0; k<6; k++)
        {
          acc_vec.val[k] <<= TENSOR_WIDTH;
          acc_vec.val[k](TENSOR_WIDTH-1, 0) = acc.val[k](TENSOR_WIDTH-1, 0);
        }
      }

      if(c>=1)
      {
        tensor[r][c-1] = acc_vec;
      }
    }
  }
}

// compute output flow
void flow_calc(tensor_vec_t tensors[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
               velocity_vec_t outputs[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR])
{
  #pragma HLS data_pack variable=tensors
  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    FLOW_INNER: for(int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1
      tensor_vec_t tensor_vec = tensors[r][c];
      vel_pixel_vec_t res_vec_x;
      vel_pixel_vec_t res_vec_y;
      for(int p=0; p<PAR_FACTOR; p++)
      {
        int cc = c * PAR_FACTOR + p;
        tensor_t tensor;
        vel_pixel_t res_x = 0;
        vel_pixel_t res_y = 0;
        for (int i=0; i<6; i++)
        {
          tensor.val[i](TENSOR_WIDTH-1, 0) = tensor_vec.val[i](TENSOR_WIDTH-1, 0);
          tensor_vec.val[i] >>= TENSOR_WIDTH;
        }
        if(r>=2 && r<MAX_HEIGHT-2 && cc>=2 && cc<MAX_WIDTH-2)
        {
          tensor_pixel_t t1 = tensor.val[0];
          tensor_pixel_t t2 = tensor.val[1];
          tensor_pixel_t t3 = tensor.val[2];
          tensor_pixel_t t4 = tensor.val[3];
          tensor_pixel_t t5 = tensor.val[4];
          tensor_pixel_t t6 = tensor.val[5];

          denom_t denom = t1*t2-t4*t4;

          if(denom != 0)
          {
            nom_t nom_x = t6*t4-t5*t2;
            nom_t nom_y = t5*t4-t6*t1;
            res_x = nom_x / denom;
            res_y = nom_y / denom;
          }
        }

        res_vec_x >>= VEL_WIDTH;
        res_vec_y >>= VEL_WIDTH;

        int msb = PAR_FACTOR*VEL_WIDTH-1;
        res_vec_x(msb, msb-VEL_WIDTH+1) = res_x(VEL_WIDTH-1, 0);
        res_vec_y(msb, msb-VEL_WIDTH+1) = res_y(VEL_WIDTH-1, 0);
      }

      outputs[r][c].x = res_vec_x;
      outputs[r][c].y = res_vec_y;
    }
  }
}

void stream_out(velocity_vec_t velocities[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR],
                axi_bus_t output[OUTPUT_LENGTH])
{
  for (int r=0; r<MAX_HEIGHT; r++)
    for (int c=0; c<MAX_WIDTH/PAR_FACTOR; c++)
    {
      #pragma HLS pipeline II=1
      velocity_vec_t vec = velocities[r][c];
  
      axi_bus_t output_word;
      for (int p=0; p<PAR_FACTOR; p++)
      {
        int msb = 2*VEL_WIDTH*PAR_FACTOR-1;
        output_word >>= VEL_WIDTH;
        output_word(msb, msb-VEL_WIDTH+1) = vec.x(VEL_WIDTH-1, 0);
        output_word >>= VEL_WIDTH;
        output_word(msb, msb-VEL_WIDTH+1) = vec.y(VEL_WIDTH-1, 0);
        vec.x >>= VEL_WIDTH;
        vec.y >>= VEL_WIDTH;
      }
      output[r * MAX_WIDTH/PAR_FACTOR + c] = output_word;
    }
}

// top-level kernel function
void optical_flow(axi_bus_t input[INPUT_LENGTH],
                  axi_bus_t output[OUTPUT_LENGTH])
{
  #pragma HLS DATAFLOW

  // FIFOs connecting the stages
  STATIC pixel_vec_t gradient_x[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=gradient_x depth=default_depth
  STATIC pixel_vec_t gradient_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=gradient_y depth=default_depth
  STATIC pixel_vec_t gradient_z[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=gradient_z depth=max_width*4
  STATIC gradient_vec_t y_filtered[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=y_filtered depth=default_depth
  STATIC gradient_vec_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=filtered_gradient depth=default_depth
  STATIC outer_vec_t out_product[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=out_product depth=default_depth
  #pragma HLS data_pack variable=out_product
  STATIC tensor_vec_t tensor_y[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=tensor_y depth=default_depth
  #pragma HLS data_pack variable=tensor_y
  STATIC tensor_vec_t tensor[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=tensor depth=default_depth
  #pragma HLS data_pack variable=tensor
  STATIC velocity_vec_t velocities[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=velocities depth=default_depth
  #pragma HLS data_pack variable=velocities

  // FIFOs for streaming in, just for clarity
  STATIC input_vec_t frame1_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame1_a depth=default_depth
  STATIC input_vec_t frame2_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame2_a depth=default_depth
  STATIC input_vec_t frame4_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame4_a depth=default_depth
  STATIC input_vec_t frame5_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame5_a depth=default_depth

  //Need to duplicate frame3 for the two calculations
  STATIC input_vec_t frame3_a[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame3_a depth=default_depth
  STATIC input_vec_t frame3_b[MAX_HEIGHT][MAX_WIDTH/PAR_FACTOR];
  #pragma HLS STREAM variable=frame3_b depth=default_depth

  // stream in and organize the inputs
  stream_in(input, frame1_a, frame2_a, frame3_a, frame3_b, frame4_a, frame5_a);
  
  // compute
  print_frame<8>("frame3_a", frame3_a);
  gradient_xy_calc(frame3_a, gradient_x, gradient_y);
  print_frame<PIX_WIDTH>("gradient_x", gradient_x);
  print_frame<PIX_WIDTH>("gradient_y", gradient_y);
  gradient_z_calc(frame1_a, frame2_a, frame3_b, frame4_a, frame5_a, gradient_z);
  print_frame<PIX_WIDTH>("gradient_z", gradient_z);
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
  flow_calc(tensor, velocities);
  print_frame("velocities", velocities);

  // stream out
  stream_out(velocities, output);
}
