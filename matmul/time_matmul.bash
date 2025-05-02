total_thread_time=0
total_block_time=0
total_device_time=0

nvcc block_matmul.cu -o block_matmul
nvcc device_matmul.cu -o device_matmul

for width in 512 1024 2048 4096
do
	echo "Running matmul for size $width x $width"
	for i in {1..100}
	do
		# extract block time from output: Time taken by shared memory KERNEL to execute is: 0.199264 ms
		block_output=$(./block_matmul $width)
		block_time=$(echo "$block_output" | grep 'Time taken by shared memory KERNEL to execute is:' | sed 's/.*is: //; s/ ms//')
		total_block_time=$(echo "$total_block_time + $block_time" | bc)

		# extract thread time from output: Time taken by naive KERNEL to execute is: 1.468672 ms
		thread_time=$(echo "$block_output" | grep 'Time taken by naive KERNEL to execute is:' | sed 's/.*is: //; s/ ms//')
		total_thread_time=$(echo "$total_thread_time + $thread_time" | bc)

		# extract device time from output: Time taken by device-level tiled streams: 2.297856 ms
		device_output=$(./device_matmul $width)
		device_time=$(echo "$device_output" | grep 'Time taken by device-level tiled streams:' | sed 's/.*: //; s/ ms//')
		total_device_time=$(echo "$total_device_time + $device_time" | bc)
	done

	# Calculate averages
	avg_block_time=$(echo "scale=6; $total_block_time / 100" | bc)
	avg_thread_time=$(echo "scale=6; $total_thread_time / 100" | bc)
	avg_device_time=$(echo "scale=6; $total_device_time / 100" | bc)

	# Print results
	echo "Average matmul ($width) thread time (ms): $avg_thread_time"
	echo "Average matmul ($width) block time (ms): $avg_block_time"
	echo "Average matmul ($width) device time (ms): $avg_device_time"

	# Reset totals for next size
	total_thread_time=0
	total_block_time=0
	total_device_time=0

	echo ""
done