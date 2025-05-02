nvcc timings_device.cu -o timings_device

# Run the kernel 100 times and average timings
total_time=0
for i in {1..100}
do
	# Run the kernel and capture the output
	time=$(./timings_device)
	# echo "Run $i: $time ms"
	
	# Add to total time
	total_time=$(echo "$total_time + $time" | bc)
done

# Calculate average time
average_time=$(echo "$total_time / 100" | bc -l)
echo "Average time: $average_time ms"