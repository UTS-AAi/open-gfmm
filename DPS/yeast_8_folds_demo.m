data_file = "yeast.dat";

M = dlmread(data_file);
r = size(M, 1);
c = size(M, 2);
A = M(:, 1:c-1);
LABS = M(:, c);
LEVELS = 3;

M = [A, LABS];
[R, H] = dps(A, LEVELS, LABS);

max_fold = 2^LEVELS;
Fold0 = M(R == 0, :);
Fold1 = M(R == 1, :);
Fold2 = M(R == 2, :);
Fold3 = M(R == 3, :);
Fold4 = M(R == 4, :);
Fold5 = M(R == 5, :);
Fold6 = M(R == 6, :);
Fold7 = M(R == 7, :);
Fold8 = M(R == 8, :);

[filepath, name, ext] = fileparts(data_file);
file_fold0 = name + "_dps_remaining_samples.dat";
file_fold1 = name + "_dps_1.dat";
file_fold2 = name + "_dps_2.dat";
file_fold3 = name + "_dps_3.dat";
file_fold4 = name + "_dps_4.dat";
file_fold5 = name + "_dps_5.dat";
file_fold6 = name + "_dps_6.dat";
file_fold7 = name + "_dps_7.dat";
file_fold8 = name + "_dps_8.dat";

dlmwrite(file_fold0, Fold0);
dlmwrite(file_fold1, Fold1);
dlmwrite(file_fold2, Fold2);
dlmwrite(file_fold3, Fold3);
dlmwrite(file_fold4, Fold4);
dlmwrite(file_fold5, Fold5);
dlmwrite(file_fold6, Fold6);
dlmwrite(file_fold7, Fold7);
dlmwrite(file_fold8, Fold8);