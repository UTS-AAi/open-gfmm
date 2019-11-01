%list_file = ["segmentation", "circle", "complex9", "DiagnosticBreastCancer", "glass", "heart", "ionosphere", "iris", "ringnorm", "spherical_5_2", "spiral", "thyroid", "twonorm", "waveform", "wine", "yeast", "zelnik6"];
list_file = ["musk_v2"];
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
    Tr = [];
    for i = [0, 2, 3, 4]
       Tr_tmp = M(R == i, :);
       Tr = [Tr; Tr_tmp];
    end
    % Val = [];
    % for i = [4]
    %    Val_tmp = M(R == i, :);
    %    Val = [Val; Val_tmp];
    % end
    Test = [];
    for i = [1]
       Test_tmp = M(R == i, :);
       Test = [Test; Test_tmp];
    end

    [filepath,name,ext] = fileparts(data_file);
    training_file = "dps_train_test/" + name + "_dps_3_train.dat";
    testing_file = "dps_train_test/" + name + "_dps_3_test.dat";
    %val_file = "dps_train_test/" + name + "_dps_val.dat";
    dlmwrite(training_file, Tr);
    dlmwrite(testing_file, Test);
    %dlmwrite(val_file, Val);
end