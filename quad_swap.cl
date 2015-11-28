float dist(uint4 x, uint4 y) {

	float LUM_WEIGHT = 10;
	float F[3] = {0.114f, 0.587f, 0.299f};

        float t = 0;
        float lx = 0;
        float ly = 0;
	for (int i = 0; i < 3; ++i) {
		float xi = x[i] * F[i];
		float yi = y[i] * F[i];
        	float d = xi - yi;
        	t += d * d;
        	lx += xi;
        	ly += yi;
	}
	float l = lx - ly;
	return t + l * l * LUM_WEIGHT;
}

int conditional_swap(__read_only image2d_t source,
                     sampler_t sampler,
                     int2 loc_a,
                     int2 loc_b,
                     int width,
                     uint4 *buffer)
{
	uint4 src_a = (read_imageui(source, sampler, loc_a));
	uint4 src_b = (read_imageui(source, sampler, loc_b));
	int pos_c = loc_a[0] * width + loc_a[1];
	int pos_d = loc_b[0] * width + loc_b[1];
	uint4 buf_c = buffer[pos_c];
	uint4 buf_d = buffer[pos_d];

	float stay_dist = dist(src_a, buf_c) + dist(src_b, buf_d);
	float swap_dist = dist(src_a, buf_d) + dist(src_b, buf_c);

	int did_swap = 0;
	if (swap_dist < stay_dist) {
		buffer[pos_c] = buf_d;
		buffer[pos_d] = buf_c;
		did_swap = 1;
	}

	return did_swap;
}

int even_horizontal(__read_only image2d_t source,
                    sampler_t sampler,
                    uint4 *buffer,
                    int width,
                    int height,
                    int gid)
{
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (row, col);
	int2 b = (int2) (row, col + 1);// Assumes even width

	return conditional_swap(source, sampler, a, b, width, buffer);
}

int odd_horizontal(__read_only image2d_t source,
                    sampler_t sampler,
                    uint4 *buffer,
                    int width,
                    int height,
                    int gid)
{
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (row, col + 1); // Assumes even width
	int2 b = (int2) (row, (col + 2) % width); // Wrap around to col 0

	return conditional_swap(source, sampler, a, b, width, buffer);
}

int even_vertical(__read_only image2d_t source,
                  sampler_t sampler,
                  uint4 *buffer,
                  int width,
                  int height,
                  int gid)
{
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (row, col);
	int2 b = (int2) (row + 1, col);// Assumes even height

	return conditional_swap(source, sampler, a, b, width, buffer);
}

int odd_vertical(__read_only image2d_t source,
                 sampler_t sampler,
                 uint4 *buffer,
                 int width,
                 int height,
                 int gid)
{
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (row + 1, col); // Assumes even height
	int2 b = (int2) ((row + 2) % height, col); // Wrap around to row 0

	return conditional_swap(source, sampler, a, b, width, buffer);
}

__kernel void quad_swap(__read_only image2d_t source,
                        __read_only image2d_t palette,
                        uint4 *buffer,
                        __write_only image2d_t destination,
                        sampler_t sampler,
                        int width,
                        int height)
{

	int gid = get_global_id(0);

	int row = (2*gid) / width;
	int col = (2*gid) % width;

	buffer[2*gid]   = read_imageui(palette, sampler, (int2)(row,col));
	buffer[2*gid+1] = read_imageui(palette, sampler, (int2)(row,col+1));

	int swaps = 0;

	for (int i = 0; i < 1000; i++) {
		//write_imageui(destination, (int2)(row,col), buffer[2*gid]);
		//write_imageui(destination, (int2)(row,col+1), buffer[2*gid+1]);

		swaps = even_horizontal(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(row,col), buffer[2*gid]);
		//write_imageui(destination, (int2)(row,col+1), buffer[2*gid+1]);

		swaps += even_vertical(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(row,col), buffer[2*gid]);
		//write_imageui(destination, (int2)(row,col+1), buffer[2*gid+1]);

		swaps += odd_horizontal(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(row,col), buffer[2*gid]);
		//write_imageui(destination, (int2)(row,col+1), buffer[2*gid+1]);

		swaps += odd_vertical(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

	}

	write_imageui(destination, (int2)(row,col), buffer[2*gid]);
	write_imageui(destination, (int2)(row,col+1), buffer[2*gid+1]);
}