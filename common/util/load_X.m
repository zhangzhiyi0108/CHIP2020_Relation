function X = load_X(data_path)

fp = fopen(data_path);
nsample = fread(fp, 1, 'int');
ndim = fread(fp, 1, 'int');
X = fread(fp, nsample * ndim, 'double');
fclose(fp);

X = reshape(X, [ndim,nsample]);
X = X';
