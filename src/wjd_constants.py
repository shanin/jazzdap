WDB_SIZE = 456

MISTAKES = {
    'DickieWells_Jo-Jo_Orig': 'DickieWells_Jo=Jo_Orig',
    "BobBerg_ IDidn'tKnowWhatTimeItWas_Orig": "BobBerg_IDidn'tKnowWhatTimeItWas_Orig",
    "Bob Berg_SecondSightEnterTheSpirit_Orig": "BobBerg_SecondSightEnterTheSpirit_Orig",
    "DexterGordon_Society Red_Orig": "DexterGordon_SocietyRed_Orig",
    "DizzyGillespie_Be-Bop_Orig": "DizzyGillespie_Be=Bop_Orig",
    "Don Byas_OutOfNowhere_Orig": "DonByas_OutOfNowhere_Orig",
    "JohnColtrane_26-2_Orig": "JohnColtrane_26=2_Orig",
    "SonnyStitt_BluesInBe-Bop_Orig": "SonnyStitt_BluesInBe=Bop_Orig",
    "LesterYoung_D.B. Blues_Orig": "LesterYoung_D.B.Blues_Orig",
    "MilesDavis_Eighty-One_Orig": "MilesDavis_Eighty=One_Orig",
    "Dave Holland_TakeTheColtrane_Orig": "DaveHolland_TakeTheColtrane_Orig",
    "ChrisPotter_InASentimentalMood_Orig": "ChrisPotter_InaSentimentalMood_Orig",
    "MilesDavis_Gingerbreadboy_Orig": "MilesDavis_GingerbreadBoy_Orig" 
}

SOLO_MISTAKES = {
    "FatsNavarro_GoodBait_No1_Solo": "FatsNavarro_GoodBait_Solo",
    "FatsNavarro_GoodBait_No2_Solo": "FatsNavarro_GoodBait_AlternateTake_Solo",
    "BranfordMarsalis_Ummg_Solo": "BranfordMarsalis_U.M.M.G._Solo",
    "SonnyRollins_I'llRememberApril-AlternateTake2_Solo": "SonnyRollins_I'llRememberApril_AlternateTake2_Solo",
    "PaulDesmond_BlueRondoAlaTurk_Solo": "PaulDesmond_BlueRondoALaTurk_Solo",
    "BranfordMarsalis_GutbucketSteepy_Solo": "BranfordMarsalis_GutBucketSteepy_Solo",
    "DizzyGillespie_Blue'NBoogie_Solo": "DizzyGillespie_Blue'nBoogie_Solo",
    "EricDolphy_Aisha_solo": "EricDolphy_Aisha_Solo",
    "KidOry_Who'sit_Solo": "KidOry_Who'sIt_Solo",
    "WayneShorter_JuJu_Solo": "WayneShorter_Juju_Solo"
}

SOLO_PATCH_FILES = ['LouisArmstrong_CornetChopSuey_Solo']

SOLOSTART_CORRECTIONS = {
    43: 29.9,
    64: -15.345,
    79: 93.1722,
    82: -93.1722,
    171: -64.85,
    309: -212.5,
    382: 205.9
}

TRAIN_ARTISTS = [
    'Art Pepper', 'Benny Carter', 'Benny Goodman', 'Bix Beiderbecke', 
    'Bob Berg', 'Branford Marsalis', 'Buck Clayton', 'Charlie Parker', 
    'Chet Baker', 'Clifford Brown', 'Coleman Hawkins', 'Curtis Fuller', 
    'David Liebman', 'David Murray', 'Dickie Wells', 'Dizzy Gillespie', 
    'Don Byas', 'Don Ellis', 'Eric Dolphy', 'Freddie Hubbard', 
    'Gerry Mulligan', 'Hank Mobley', 'Harry Edison', 
    'Henry Allen', 'Herbie Hancock', 'J.C. Higginbotham', 'J.J. Johnson', 
    'Joe Lovano', 'John Abercrombie', 'Johnny Dodds', 'Johnny Hodges', 
    'Joshua Redman', 'Kai Winding', 'Kenny Dorham', 'Kenny Garrett', 
    'Kid Ory', 'Lee Konitz', 'Lee Morgan', 'Lester Young', 'Lionel Hampton', 
    'Louis Armstrong', 'Michael Brecker', 'Miles Davis', 'Milt Jackson', 
    'Ornette Coleman', 'Pat Martino', 'Pat Metheny', 
    'Pepper Adams', 'Red Garland', 'Rex Stewart', 
    'Roy Eldridge', 'Sonny Stitt', 'Stan Getz', 'Steve Coleman', 
    'Steve Lacy', 'Steve Turre', 'Von Freeman', 'Warne Marsh', 
    'Wayne Shorter', 'Woody Shaw', 'Zoot Sims'
]


VAL_ARTISTS = [
    'Nat Adderley', 'George Coleman', 'Phil Woods'
]

TEST_ARTISTS = [
    'Ben Webster', 'Cannonball Adderley', 'Charlie Shavers', 'Chu Berry', 
    'Chris Potter', 'Dexter Gordon', 'Fats Navarro', 'Joe Henderson', 
    'John Coltrane', 'Kenny Wheeler', 'Paul Desmond', 'Sidney Bechet', 
    'Sonny Rollins', 'Wynton Marsalis'
]

# these tracks should be omitted from test to maintain compatibility with 2019 results
TEST_STOPLIST = {
    79: "ChrisPotter_Arjuna_Solo",
    82: "ChrisPotter_PopTune#1_Solo",
    223: "JohnColtrane_GiantSteps-2_Solo",
    259: "KennyWheeler_PassItOn_Solo",
    382: "SonnyRollins_I'llRememberApril_AlternateTake2_Solo"
}


INSTRUMENTS = [
    'cl', 'as', 'ts', 'cor', 'tp', 'tb', 'ss', 'bcl', 'bs', 'p',
    'ts-c', 'g', 'vib'
]

MELODY_COLNAMES = [
    'eventid', 'melid', 'onset', 'pitch', 'duration', 'period', 
    'division', 'bar', 'beat', 'tatum', 'subtatum', 'num', 'denom', 
    'beatprops', 'beatdur', 'tatumprops', 'loud_max', 'loud_med', 
    'loud_sd', 'loud_relpos', 'loud_cent', 'loud_s2b', 'f0_mod', 
    'f0_range', 'f0_freq_hz', 'f0_med_dev'
]

SOLO_INFO_COLNAMES = [
    'melid', 'trackid', 'compid', 'recordid', 'performer', 'title', 
    'titleaddon', 'solopart', 'instrument', 'style', 'avgtempo', 
    'tempoclass', 'rhythmfeel', 'key', 'signature', 'chord_changes', 
    'chorus_count'
]

TRANSCRIPTION_INFO_COLNAMES = [
    'melid', 'trackid', 'filename_sv', 'filename_solo', 
    'solotime', 'solostart_sec', 'status'
]

TRACK_INFO_COLNAMES = [
    'trackid', 'compid', 'recordid', 'filename_track', 'lineup',
    'mbzid', 'trackno', 'recordingdate'
]

BEATS_COLNAMES = [
    'beatid', 'melid', 'onset', 'bar', 'beat', 'signature', 'chord',
    'form', 'bass_pitch', 'chorus_id'
]