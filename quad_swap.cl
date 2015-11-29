/**
 * Calculate pixel color distance.
 *
 * @param x a vector with RGB values in .x, .y, and .z
 * @param y a vector with RGB values in .x, .y, and .z
 *
 * @returns The color distance between x and y;
 */
float dist(uint4 x, uint4 y) {

	float LUM_WEIGHT = 10;
	float F[3] = {0.114f, 0.587f, 0.299f};

        float t = 0;
        float lx = 0;
        float ly = 0;

	float xi[3];
	float yi[3];
	xi[0] = x.x * F[0];
	xi[1] = x.y * F[1];
	xi[2] = x.z * F[2];
	yi[0] = y.x * F[0];
	yi[1] = y.y * F[1];
	yi[2] = y.z * F[2];
	lx = xi[0] + xi[1] + xi[2];
	ly = yi[0] + yi[1] + yi[2];
	xi[0] -= yi[0];
	xi[1] -= yi[1];
	xi[2] -= yi[2];
	t = (xi[0] * xi[0]) +
	    (xi[1] * xi[1]) +
	    (xi[2] * xi[2]);
	float l = lx - ly;
	return t + l * l * LUM_WEIGHT;
}

/**
 * Swap two pixels if swapping them would lower the total distance between
 * pixels in @a source and the corresponding pixels in @a buffer.
 *
 * @param source  The source image.
 * @param sampler A sampler for @a source.
 * @param loc_a   The coordinates (column, row) of the first pixel.
 * @param loc_b   The coordinates (column, row) of the second pixel.
 * @param width   The width in pixels of @a source.
 * @param buffer  The workspace where pixels at @a loc_a and @a loc_b may
 *                be swapped.
 */
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

/**
 * Do a conditional, horizontal swap with pixels on an "even" cut.
 *
 * @param source  The source image.
 * @param sampler A sampler for @a source.
 * @param buffer  The workspace in which to swap pixels.
 * @param width   The width of @a source.
 * @param height  The height of @a source.
 * @param gid     The global ID inherited from the kernel.
 *
 * An "even" cut means that the pixels will be swapped in pairs starting at
 * index 0. That is, for a 4x4 image, a horizontal swap of all pixels would be:
 *
 * a b c d   ->   b a d c
 * e f g h   ->   f e h g
 * i j k l   ->   j i l k
 * m n o p   ->   n m p o
 */
int even_horizontal(__read_only image2d_t source,
                    sampler_t sampler,
                    uint4 *buffer,
                    int width,
                    int height,
                    int gid)
{
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (col, row);
	int2 b = (int2) (col + 1, row);// Assumes even width

	return conditional_swap(source, sampler, a, b, width, buffer);
}

/**
 * Do a conditional, horizontal swap with pixels on an "odd" cut.
 *
 * @param source  The source image.
 * @param sampler A sampler for @a source.
 * @param buffer  The workspace in which to swap pixels.
 * @param width   The width of @a source.
 * @param height  The height of @a source.
 * @param gid     The global ID inherited from the kernel.
 *
 * An "odd" cut means that the pixels will be swapped in pairs starting at
 * index 1 with wrap around to 0 when pairing the final pixel in each row.
 * That is, for a 4x4 image, a horizontal swap of all pixels would be:
 *
 * a b c d   ->   d c b a
 * e f g h   ->   h g f e
 * i j k l   ->   l k j i
 * m n o p   ->   p o n m
 */
int odd_horizontal(__read_only image2d_t source,
                    sampler_t sampler,
                    uint4 *buffer,
                    int width,
                    int height,
                    int gid)
{
	int row = (2 * gid) / width;
	int col = (2 * gid) % width;

	int2 a = (int2) (col + 1, row); // Assumes even width
	int2 b = (int2) ((col + 2) % width, row); // Wrap around to col 0

	return conditional_swap(source, sampler, a, b, width, buffer);
}

/**
 * Do a conditional, vertical swap with pixels on an "even" cut.
 *
 * @param source  The source image.
 * @param sampler A sampler for @a source.
 * @param buffer  The workspace in which to swap pixels.
 * @param width   The width of @a source.
 * @param height  The height of @a source.
 * @param gid     The global ID inherited from the kernel.
 *
 * An "even" cut means that the pixels will be swapped in pairs starting at
 * index 0. That is, for a 4x4 image, a vertical swap of all pixels would be:
 *
 * a b c d   ->   e f g h
 * e f g h   ->   a b c d
 * i j k l   ->   m n o p
 * m n o p   ->   i j k l
 */
int even_vertical(__read_only image2d_t source,
                  sampler_t sampler,
                  uint4 *buffer,
                  int width,
                  int height,
                  int gid)
{
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (col, row);
	int2 b = (int2) (col, row + 1);// Assumes even height

	return conditional_swap(source, sampler, a, b, width, buffer);
}

/**
 * Do a conditional, vertical swap with pixels on an "odd" cut.
 *
 * @param source  The source image.
 * @param sampler A sampler for @a source.
 * @param buffer  The workspace in which to swap pixels.
 * @param width   The width of @a source.
 * @param height  The height of @a source.
 * @param gid     The global ID inherited from the kernel.
 *
 * An "odd" cut means that the pixels will be swapped in pairs starting at
 * index 1 with wrap around to 0 when pairing the final pixel in each column.
 * That is, for a 4x4 image, a vertical swap of all pixels would be:
 *
 * a b c d   ->   m n o p
 * e f g h   ->   i j k l
 * i j k l   ->   e f g h
 * m n o p   ->   a b c d
 */
int odd_vertical(__read_only image2d_t source,
                 sampler_t sampler,
                 uint4 *buffer,
                 int width,
                 int height,
                 int gid)
{
	int row = 2 * (gid / width);
	int col = gid % width;

	int2 a = (int2) (col, row + 1); // Assumes even height
	int2 b = (int2) (col, (row + 2) % height); // Wrap around to row 0

	return conditional_swap(source, sampler, a, b, width, buffer);
}

/**
 * Iteratively swap pixels from a palette to approximate the source image.
 *
 * @param source      The source image.
 * @param palette     The palette image.
 * @param buffer      A workspace to place intermediate results.
 * @param destination A write-only image to place final (or current) state
 *                    of @a buffer.
 * @param sampler     A sampler for all three image arguments.
 * @param width       The width of @a source.
 * @param height      The height of @a source.
 */
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

	buffer[2*gid]   = read_imageui(palette, sampler, (int2)(col, row));
	buffer[2*gid+1] = read_imageui(palette, sampler, (int2)(col+1, row));

	int swaps = 0;

	for (int i = 0; i < 1000; i++) {
		//write_imageui(destination, (int2)(col, row), buffer[2*gid]);
		//write_imageui(destination, (int2)(col+1, row), buffer[2*gid+1]);

		swaps = even_horizontal(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(col, row), buffer[2*gid]);
		//write_imageui(destination, (int2)(col+1, row), buffer[2*gid+1]);

		swaps += even_vertical(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(col, row), buffer[2*gid]);
		//write_imageui(destination, (int2)(col+1, row), buffer[2*gid+1]);

		swaps += odd_horizontal(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

		//write_imageui(destination, (int2)(col, row), buffer[2*gid]);
		//write_imageui(destination, (int2)(col+1, row), buffer[2*gid+1]);

		swaps += odd_vertical(source, sampler, buffer, width, height, gid);
		barrier(CLK_GLOBAL_MEM_FENCE);

	}

	write_imageui(destination, (int2)(col, row), buffer[2*gid]);
	write_imageui(destination, (int2)(col+1, row), buffer[2*gid+1]);
}
