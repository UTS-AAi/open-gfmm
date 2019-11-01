%list_file = ["segmentation", "circle", "complex9", "DiagnosticBreastCancer", "glass", "heart", "ionosphere", "iris", "ringnorm", "spherical_5_2", "spiral", "thyroid", "twonorm", "waveform", "wine", "yeast", "zelnik6"];
list_file = ["vowel"];
for f = 1:length(list_file)
    data_file = strcat('original_data/', list_file(f), '.dat');
    M = dlmread(data_file);
    r = size(M, 1);
    c = size(M, 2);
    A = M(:, 1:c-1);
    LABS = M(:, c);
    LEVELS = 2;
    A = Normalise(A, [0, 1]);
    M = [A, LABS];
    [R, H] = dps(A, LEVELS, LABS);

    max_fold = 2^LEVELS;
    Fold0 = M(R == 0, :);
    Fold1 = M(R == 1, :);
    Fold2 = M(R == 2, :);
    Fold3 = M(R == 3, :);
    Fold4 = M(R == 4, :);
    %Fold5 = M(R == 5, :);
    %Fold6 = M(R == 6, :);
    %Fold7 = M(R == 7, :);
    %Fold8 = M(R == 8, :);
  
    [filepath,name,ext] = fileparts(data_file);
    file_fold0 = "dps_train_test/" + name + "_dps_0.dat";
    file_fold1 = "dps_train_test/" + name + "_dps_1.dat";
    file_fold2 = "dps_train_test/" + name + "_dps_2.dat";
    file_fold3 = "dps_train_test/" + name + "_dps_3.dat";
    file_fold4 = "dps_train_test/" + name + "_dps_4.dat";
    %file_fold5 = "dps_train_test/" + name + "_dps_5.dat";
    %file_fold6 = "dps_train_test/" + name + "_dps_6.dat";
    %file_fold7 = "dps_train_test/" + name + "_dps_7.dat";
    %file_fold8 = "dps_train_test/" + name + "_dps_8.dat";
    
    dlmwrite(file_fold0, Fold0);
    dlmwrite(file_fold1, Fold1);
    dlmwrite(file_fold2, Fold2);
    dlmwrite(file_fold3, Fold3);
    dlmwrite(file_fold4, Fold4);
    %dlmwrite(file_fold5, Fold5);
    %dlmwrite(file_fold6, Fold6);
    %dlmwrite(file_fold7, Fold7);
    %dlmwrite(file_fold8, Fold8);
end