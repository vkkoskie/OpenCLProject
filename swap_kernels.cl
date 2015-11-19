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

void conditional_swap(__read_only image2d_t source,
                      __read_only image2d_t palette,
                      __write_only image2d_t destination,
                      sampler_t sampler,
                      int2 loc_a,
                      int2 loc_b)
{
	uint4 src_a = (read_imageui(source, sampler, loc_a));
	uint4 src_b = (read_imageui(source, sampler, loc_b));
	uint4 pal_a = (read_imageui(palette, sampler, loc_a));
	uint4 pal_b = (read_imageui(palette, sampler, loc_b));

	float stay_dist = dist(src_a, pal_a) + dist(src_b, pal_b);
	float swap_dist = dist(src_a, pal_b) + dist(src_b, pal_a);

	if (swap_dist < stay_dist) {
		int2 tmp = loc_a;
		loc_a = loc_b;
		loc_b = tmp;
	}
	write_imageui(destination, loc_a, pal_a);
	write_imageui(destination, loc_b, pal_b);
}

__kernel void even_horizontal(__read_only image2d_t source,
                              __read_only image2d_t palette,
                              __write_only image2d_t destination,
                              sampler_t sampler,
                              int width,
                              int height)
{
	int gid = get_global_id(0);
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (row, col);
	int2 b = (int2) (row, col + 1);// Assumes even width

	conditional_swap(source, palette, destination, sampler, a, b);
}

__kernel void odd_horizontal(__read_only image2d_t source,
                             __read_only image2d_t palette,
                             __write_only image2d_t destination,
                             sampler_t sampler,
                             int width,
                             int height)
{
	int gid = get_global_id(0);
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (row, col + 1); // Assumes even width
	int2 b = (int2) (row, (col + 2) % width); // Wrap around to col 0

	conditional_swap(source, palette, destination, sampler, a, b);
}

__kernel void even_vertical(__read_only image2d_t source,
                            __read_only image2d_t palette,
                            __write_only image2d_t destination,
                            sampler_t sampler,
                            int width,
                            int height)
{
	int gid = get_global_id(0);
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (row, col);
	int2 b = (int2) (row + 1, col);// Assumes even height

	conditional_swap(source, palette, destination, sampler, a, b);
}

__kernel void odd_vertical(__read_only image2d_t source,
                           __read_only image2d_t palette,
                           __write_only image2d_t destination,
                           sampler_t sampler,
                           int width,
                           int height)
{
	int gid = get_global_id(0);
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (row + 1, col); // Assumes even height
	int2 b = (int2) ((row + 2) % height, col); // Wrap around to row 0

	conditional_swap(source, palette, destination, sampler, a, b);
}

