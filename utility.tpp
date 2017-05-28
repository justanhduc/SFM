template <typename T> int sign(T val) {
	return val >= 0 ? 1 : -1;
}


template <typename T> int getMaxIdx(T* arr, int size) {
	int maxIdx;
	double max = 0;
	for (int i = 0; i < size; ++i) {
		if ((double)arr[i] > max)
			maxIdx = i;
	}
	return maxIdx;
}
