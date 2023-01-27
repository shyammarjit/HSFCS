def get_optimal_channels(subject):
	if(subject=='s01'):
		val = ['Fz', 'CP2', 'P8', 'CP1', 'O2', 'Pz', 'P7']
		ar = ['P7', 'Fp2', 'AF4', 'F8', 'Fp1']
		four = ['P7', 'P8', 'CP6', 'F7', 'T8', 'FC5', 'C3', 'AF3', 'Fp1', 'Cz']
		return val, ar, four

	if(subject=='s02'):
		val = ['Fp2', 'Cz', 'Oz', 'F4', 'FC1', 'CP2', 'O1', 'FC5', 'Fp1', 'F7', 'C3', 'AF4', 'FC6', 'CP5']
		ar = ['P7', 'F3', 'FC2', 'Pz', 'C4', 'CP2', 'T8', 'P3', 'FC5', 'F8', 'P4', 'CP5', 'CP1', 'O2', 'FC6', 'Cz', 'PO3', 'P8', 'O1', 'CP6', 'AF4', 'Oz', 'F4']
		four = ['P7', 'F4', 'F3', 'T7', 'O1', 'C4', 'Fz', 'T8', 'Fp1', 'P4', 'CP6', 'CP5', 'CP1', 'C3', 'PO3', 'O2']
		return val, ar, four
	
	if(subject=='s03'):
		val = ['O2', 'F7', 'P8']
		ar = ['AF4', 'Fp1']
		four = ['O2']
		return val, ar, four
	
	if(subject=='s04'):
		val = ['CP6', 'C3', 'FC5', 'PO3', 'FC6', 'AF4', 'Fz', 'FC1', 'Oz', 'AF3', 'P8', 'Fp1', 'F7', 'F4', 'PO4', 'Fp2', 'Pz', 'P4', 'O1', 'CP5']
		ar = ['T8', 'FC2', 'Fz', 'Pz', 'CP6', 'P3']
		four = ['FC2', 'T8', 'Pz', 'F3', 'Fz', 'PO3', 'C3', 'CP6', 'FC5', 'AF4', 'FC1', 'FC6', 'CP2', 'F4', 'F7', 'CP1', 'O1', 'Fp2', 'P4', 'PO4', 'C4', 'CP5', 'Cz', 'T7']
		return val, ar, four
	
	if(subject=='s05'):
		val = ['C4', 'F3', 'Fp1', 'CP6', 'C3', 'Cz', 'CP5', 'F7']
		ar = ['PO3', 'Pz', 'P3', 'CP5', 'C3', 'Fp2', 'FC5', 'PO4', 'T7', 'F8', 'FC2', 'FC6']
		four = ['F3', 'F8', 'FC5']
		return val, ar, four
	
	if(subject=='s06'):
		val = ['P4', 'Fz', 'P3', 'Fp1', 'CP2', 'FC2', 'P7', 'C4', 'PO3', 'F3']
		ar = ['C3', 'F3', 'AF4', 'AF3', 'FC2', 'F4', 'T8', 'Oz', 'FC5', 'F7', 'FC1', 'Cz', 'Fp2', 'O2', 'PO3', 'CP6', 'PO4']
		four = ['C3', 'Fz']
		return val, ar, four
	
	if(subject=='s07'):
		val = ['T8', 'Pz', 'F7', 'CP5', 'F8', 'FC1']
		ar = ['FC1', 'O1', 'FC5', 'PO3', 'P7', 'Fp2', 'CP1', 'CP6', 'Cz']
		four = ['F3', 'FC1', 'T7', 'P7', 'F8']
		return val, ar, four
	
	if(subject=='s08'):
		val = ['Fp1', 'P8', 'P7', 'P3', 'FC2', 'AF4', 'CP1', 'CP5', 'FC1', 'CP2', 'PO4', 'T7', 'P4', 'Fp2']
		ar = ['Fz', 'AF4', 'C4', 'AF3', 'F4', 'FC1', 'Fp1', 'CP1', 'O1', 'P4', 'F3']
		four = ['Fp1', 'AF3', 'FC1']
		return val, ar, four
	
	if(subject=='s09'):
		val = ['T8', 'AF4', 'T7', 'Oz', 'Cz', 'FC1', 'CP6', 'Pz', 'F3', 'O2']
		ar = ['AF4', 'Oz', 'T8', 'T7', 'P8', 'AF3', 'Fp1', 'Fz', 'Pz', 'P3', 'CP6', 'CP1', 'FC2', 'CP2', 'P7', 'FC6', 'P4', 'CP5', 'F3', 'PO4', 'FC5', 'O2']
		four = ['AF4', 'T7', 'Oz', 'P8', 'FC6', 'FC1', 'Cz', 'CP6', 'Pz', 'P3', 'P7', 'FC2', 'CP1', 'P4', 'F3', 'CP5', 'F4', 'PO4', 'O2', 'FC5', 'F7', 'O1', 'F8']
		return val, ar, four
	
	if(subject=='s10'):
		val = ['P3', 'CP1', 'FC6', 'Cz', 'F4', 'PO4']
		ar = ['PO4', 'Fz']
		four = ['P3', 'FC6', 'CP2', 'FC2']
		return val, ar, four
	
	if(subject=='s11'):
		val = ['Fp1', 'Fp2', 'AF3', 'CP5', 'P7', 'FC2', 'FC6', 'C3', 'PO4', 'F4', 'P4', 'P8', 'O2', 'CP6', 'F7', 'FC5', 'CP1']
		ar = ['P4', 'Fz', 'Pz', 'P8', 'Oz', 'O2', 'PO3', 'F4', 'C3', 'PO4', 'O1', 'Cz', 'T8', 'C4', 'CP1', 'CP6', 'AF4', 'F7', 'CP2', 'P3', 'F8', 'FC1', 'F3', 'P7', 'FC5', 'FC2', 'FC6', 'Fp2', 'Fp1']
		four = ['F4', 'Fz', 'P8', 'Oz', 'C3', 'Cz', 'Pz', 'PO3', 'CP6', 'PO4', 'O1', 'O2', 'C4', 'P4', 'F7', 'CP1', 'T8', 'CP2', 'P3', 'FC5', 'FC1', 'F8', 'P7', 'FC2', 'F3', 'FC6', 'AF3']
		return val, ar, four
	
	if(subject=='s12'):
		val = ['CP1', 'FC2', 'F3']
		ar = ['FC6', 'F7', 'CP5', 'FC5']
		four = ['CP1', 'Cz', 'FC5', 'C4', 'O1', 'Fp1', 'P7']
		return val, ar, four
	
	if(subject=='s13'):
		val = ['P7', 'FC1', 'Cz', 'O2', 'Pz', 'Oz', 'AF3', 'F8', 'CP6', 'O1', 'F7', 'AF4', 'CP2', 'FC6', 'T7']
		ar = ['T8', 'CP1', 'P3', 'C4']
		four = ['P7', 'P8', 'FC1', 'Oz', 'O2', 'FC6', 'O1', 'P4', 'F4', 'AF3', 'CP6', 'CP1']
		return val, ar, four
	
	if(subject=='s14'):
		val = ['F7', 'O2', 'CP2', 'F8', 'CP6', 'Fz', 'AF3', 'Pz', 'P3', 'FC2']
		ar = ['AF3', 'Oz', 'FC1', 'Fp1', 'F8', 'C4', 'AF4', 'PO3']
		four = ['F7', 'T7', 'T8', 'O2', 'P8', 'Cz', 'CP2', 'Pz']
		return val, ar, four
	
	if(subject=='s15'):
		val = ['Pz', 'C3', 'FC6', 'O1', 'P4', 'T8', 'F7', 'O2', 'FC5', 'F8', 'CP2', 'PO4', 'T7', 'AF3', 'C4', 'CP6', 'P8', 'F4', 'Fp1', 'Fp2', 'CP5', 'P3', 'FC1', 'Cz', 'P7']
		ar = ['FC1', 'P3', 'FC2', 'CP6', 'P8', 'O1', 'C3', 'C4']
		four = ['FC6', 'P4', 'O2', 'CP6', 'F8', 'F3', 'T7', 'FC2', 'FC1', 'AF3', 'AF4']
		return val, ar, four
	
	if(subject=='s16'):
		val = ['T7', 'F3', 'Oz', 'PO4', 'FC6', 'PO3', 'Fp1', 'T8', 'O1', 'P3', 'P8', 'CP6', 'CP2', 'Fp2', 'Cz']
		ar = ['O2', 'FC1', 'P8', 'Fz', 'Cz']
		four = ['O2', 'FC6', 'T7', 'FC1', 'T8', 'PO4', 'CP6', 'C3', 'P3', 'P4', 'P7', 'CP5', 'Fp2', 'C4', 'F8']
		return val, ar, four
	
	if(subject=='s17'):
		val = ['CP2', 'P3', 'FC6', 'PO3', 'F3', 'FC5', 'C4', 'CP1', 'T7', 'PO4', 'FC1']
		ar = ['P4', 'Pz', 'PO4', 'O1', 'PO3', 'C4', 'C3']
		four = ['P4', 'CP1', 'P8', 'AF4', 'FC2', 'F3', 'Pz', 'PO4', 'AF3', 'Oz', 'O1', 'F4', 'T7', 'F8']
		return val, ar, four
	
	if(subject=='s18'):
		val = ['Oz', 'Pz', 'FC1', 'F8', 'P4']
		ar = ['FC5', 'AF4', 'C4', 'PO3']	
		four = ['Oz', 'FC6', 'FC1']
		return val, ar, four
	
	if(subject=='s19'):
		val = ['F3', 'FC1', 'P7', 'Pz']
		ar = ['CP5', 'F8', 'P8', 'Oz', 'O2', 'CP6']
		four = ['O1', 'P8']
		return val, ar, four
	
	if(subject=='s20'):
		val = ['P3', 'Pz', 'P7', 'CP6', 'Fp1', 'C3', 'FC1', 'F4']
		ar = ['Fp1', 'P7', 'F3', 'AF3', 'F8', 'FC5', 'CP5', 'AF4', 'T7', 'C3']
		four = ['P7', 'P3', 'Fp1', 'F3', 'C3', 'FC1', 'T7', 'CP5']
		return val, ar, four
	
	if(subject=='s21'):
		val = ['Oz', 'Fp1', 'P3', 'F4']
		ar = ['P7', 'AF3']
		four = ['P7', 'Fz', 'AF4']
		return val, ar, four
	
	if(subject=='s22'):
		val = ['FC5', 'Fz', 'AF3', 'FC2', 'AF4', 'C3', 'Fp1', 'F4']
		ar = ['Oz', 'AF3', 'FC2', 'FC1']
		four = ['FC5', 'F3', 'AF3', 'O2', 'FC1', 'C4', 'CP5', 'Oz', 'FC2', 'C3', 'T8']
		return val, ar, four
	
	if(subject=='s23'):
		val = ['T8', 'Oz', 'PO4', 'O1', 'FC5', 'FC6', 'CP2', 'F7', 'Pz', 'O2']
		ar = ['Fp2', 'C3', 'CP5', 'CP2', 'FC2', 'FC5', 'P4', 'AF4', 'T8']
		four = ['T7', 'C3', 'CP2', 'Fz', 'F3', 'AF3']
		return val, ar, four
	
	if(subject=='s24'):
		val = ['Pz', 'PO3', 'O2', 'F8', 'FC1', 'Cz', 'T8', 'P7', 'F4', 'Fp2', 'AF4', 'CP1', 'Fp1', 'FC5']
		ar = ['O2', 'P8', 'CP2', 'CP1', 'PO3', 'FC5', 'Cz', 'FC2', 'C3', 'PO4', 'FC1', 'P7', 'T7']
		four = ['O2', 'Pz', 'O1', 'F8', 'Cz', 'T8']
		return val, ar, four
	
	if(subject=='s25'):
		val = ['F4']
		ar = ['F8', 'C4', 'PO3', 'FC5', 'AF4', 'F4', 'CP2', 'T7', 'Fp1', 'C3', 'AF3', 'Oz', 'P8', 'FC2', 'PO4', 'FC6']
		four = ['F4', 'FC5']
		return val, ar, four
	
	if(subject=='s26'):
		val = ['Pz', 'PO3', 'FC1', 'Oz']
		ar = ['CP5', 'AF4', 'F7', 'Oz', 'FC2', 'AF3', 'Cz', 'F4', 'P4', 'P3', 'PO4']
		four = ['P3', 'CP1', 'Pz', 'FC2', 'O1', 'O2', 'CP2', 'P8', 'F8', 'FC6']
		return val, ar, four
	
	if(subject=='s27'):
		val = ['Oz', 'F3', 'C4', 'AF3', 'O2', 'Fp2', 'Fp1', 'Cz']
		ar = ['Oz', 'O2', 'Pz', 'CP1', 'FC5', 'AF4', 'F8', 'CP5', 'FC2', 'P4', 'Cz', 'F4', 'Fp1']
		four = ['Oz']
		return val, ar, four
	
	if(subject=='s28'):
		val = ['C4', 'C3', 'P4']
		ar = ['PO3', 'F4', 'FC6', 'C4', 'Cz', 'CP1', 'CP5', 'C3']
		four = ['PO3', 'C4', 'P4']
		return val, ar, four
	
	if(subject=='s29'):
		val = ['F8', 'PO4', 'AF3']
		ar = ['PO4', 'P4', 'O2', 'FC2', 'F8', 'Cz', 'C4', 'CP2', 'Pz']
		four = ['PO4', 'P4', 'FC2', 'F8', 'O2', 'PO3', 'P7', 'FC6']
		return val, ar, four
	
	if(subject=='s30'):
		val = ['Cz', 'PO4', 'Fz', 'CP1', 'Pz', 'FC6', 'CP2', 'P3', 'PO3', 'AF3', 'CP5']
		ar = ['PO4', 'Fz', 'P3', 'AF3', 'P4', 'FC2', 'CP1', 'FC6', 'Pz', 'F3', 'CP2', 'CP5', 'C4', 'FC5', 'P8', 'Fp2', 'CP6', 'T7', 'T8', 'F4', 'Fp1', 'O2', 'AF4']
		four = ['PO4', 'Fz', 'Oz', 'Cz', 'FC1', 'PO3', 'CP1', 'F3', 'Fp2', 'O1', 'FC5', 'T8']
		return val, ar, four
	
	if(subject=='s31'):
		val = ['P8', 'C4', 'P4']
		ar = ['C4', 'PO4']
		four = ['C4', 'PO3', 'Oz']
		return val, ar, four
	
	if(subject=='s32'):
		val = ['CP2', 'CP6', 'O2', 'CP5', 'O1', 'P4', 'FC6', 'Oz']
		ar = ['Cz', 'AF4', 'F3']
		four = ['CP2', 'CP6', 'FC2', 'PO4', 'CP5', 'FC5', 'O2', 'P7', 'Oz', 'C4']
		return val, ar, four