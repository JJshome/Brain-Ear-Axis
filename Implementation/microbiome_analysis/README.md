# Microbiome Analysis Module (200)

This module implements the microbiome analysis component of the Brain-Ear Axis Analysis System. It focuses on analyzing the microbial communities in and around the ear, as well as exploring the auditory-gut-brain axis.

<div align="center">
  <svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
    <style>
      @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
      }
      
      @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
      
      @keyframes pulse {
        0% { r: 5; opacity: 0.5; }
        50% { r: 8; opacity: 1; }
        100% { r: 5; opacity: 0.5; }
      }
      
      @keyframes colorPulse {
        0% { fill: #2193b0; opacity: 0.7; }
        50% { fill: #6dd5ed; opacity: 0.9; }
        100% { fill: #2193b0; opacity: 0.7; }
      }
      
      @keyframes flow {
        from { stroke-dashoffset: 0; }
        to { stroke-dashoffset: -1000; }
      }
      
      .microbial-cell {
        animation: pulse 3s infinite;
      }
      
      .microbial-cluster {
        animation: float 6s ease-in-out infinite;
      }
      
      .module {
        animation: float 6s ease-in-out infinite;
      }
      
      .connection {
        stroke-dasharray: 5;
        animation: flow 30s linear infinite;
      }
      
      .rotating-microbes {
        transform-origin: center;
        animation: rotate 60s linear infinite;
      }
      
      .color-pulse {
        animation: colorPulse 4s infinite;
      }
      
      .delay-1 { animation-delay: 0.5s; }
      .delay-2 { animation-delay: 1s; }
      .delay-3 { animation-delay: 1.5s; }
      .delay-4 { animation-delay: 2s; }
      .delay-5 { animation-delay: 2.5s; }
    </style>
    
    <!-- Background with gradient -->
    <defs>
      <linearGradient id="bg-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#1b262c" />
        <stop offset="100%" stop-color="#0f4c75" />
      </linearGradient>
      
      <linearGradient id="module-gradient-1" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#2193b0" />
        <stop offset="100%" stop-color="#6dd5ed" />
      </linearGradient>
      
      <linearGradient id="module-gradient-2" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#3c8ce7" />
        <stop offset="100%" stop-color="#00eaff" />
      </linearGradient>
      
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="5" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>
    
    <rect width="100%" height="100%" fill="url(#bg-gradient)" />
    
    <!-- Title -->
    <text x="400" y="50" font-family="Arial" font-size="24" fill="white" text-anchor="middle" font-weight="bold">
      Microbiome Analysis
    </text>
    <text x="400" y="80" font-family="Arial" font-size="16" fill="rgba(255,255,255,0.8)" text-anchor="middle">
      Ear-Gut-Brain Microbial Communities
    </text>
    
    <!-- Central Ear with Microbiome -->
    <g transform="translate(400, 250)">
      <!-- Ear outline -->
      <path d="M-30,30 C-60,-45 -45,-90 0,-120 C45,-150 90,-135 120,-105 
               C150,-75 165,-30 150,30 C135,90 90,135 30,120 
               C-30,105 -60,105 -30,30 Z" 
            stroke="#2193b0" stroke-width="2" fill="rgba(33, 147, 176, 0.1)" />
      
      <!-- Ear canal -->
      <path d="M0,-90 C15,-75 30,-60 30,-30 C30,0 15,30 0,45 C-15,30 -30,0 -30,-30 C-30,-60 -15,-75 0,-90 Z" 
            stroke="#2193b0" stroke-width="1" fill="none" />
      
      <!-- Rotating Microbiome around Ear -->
      <g class="rotating-microbes">
        <!-- Microbe clusters -->
        <g class="microbial-cluster delay-1" transform="translate(-50, -50)">
          <circle cx="0" cy="0" r="12" fill="url(#module-gradient-1)" opacity="0.6" />
          <circle cx="-5" cy="-5" r="3" fill="#6dd5ed" class="microbial-cell" />
          <circle cx="5" cy="-3" r="2" fill="#6dd5ed" class="microbial-cell delay-2" />
          <circle cx="0" cy="5" r="2.5" fill="#6dd5ed" class="microbial-cell delay-3" />
          <circle cx="-3" cy="2" r="2" fill="#6dd5ed" class="microbial-cell delay-4" />
        </g>
        
        <g class="microbial-cluster delay-2" transform="translate(60, -30)">
          <circle cx="0" cy="0" r="15" fill="url(#module-gradient-1)" opacity="0.6" />
          <circle cx="-7" cy="-5" r="3" fill="#6dd5ed" class="microbial-cell delay-1" />
          <circle cx="5" cy="-8" r="4" fill="#6dd5ed" class="microbial-cell delay-3" />
          <circle cx="8" cy="5" r="3.5" fill="#6dd5ed" class="microbial-cell" />
          <circle cx="-3" cy="6" r="3" fill="#6dd5ed" class="microbial-cell delay-2" />
          <circle cx="2" cy="0" r="2" fill="#6dd5ed" class="microbial-cell delay-4" />
        </g>
        
        <g class="microbial-cluster delay-3" transform="translate(30, 70)">
          <circle cx="0" cy="0" r="10" fill="url(#module-gradient-1)" opacity="0.6" />
          <circle cx="-4" cy="-3" r="2.5" fill="#6dd5ed" class="microbial-cell delay-2" />
          <circle cx="4" cy="-2" r="2" fill="#6dd5ed" class="microbial-cell delay-4" />
          <circle cx="0" cy="4" r="3" fill="#6dd5ed" class="microbial-cell delay-1" />
        </g>
        
        <g class="microbial-cluster delay-4" transform="translate(-70, 20)">
          <circle cx="0" cy="0" r="12" fill="url(#module-gradient-1)" opacity="0.6" />
          <circle cx="-5" cy="-4" r="3" fill="#6dd5ed" class="microbial-cell delay-3" />
          <circle cx="5" cy="-5" r="2.5" fill="#6dd5ed" class="microbial-cell delay-1" />
          <circle cx="3" cy="4" r="2" fill="#6dd5ed" class="microbial-cell" />
          <circle cx="-4" cy="5" r="2.5" fill="#6dd5ed" class="microbial-cell delay-2" />
        </g>
        
        <!-- Additional microbes scattered around -->
        <circle cx="-20" cy="-80" r="7" class="color-pulse" />
        <circle cx="80" cy="-50" r="5" class="color-pulse delay-2" />
        <circle cx="90" cy="30" r="6" class="color-pulse delay-3" />
        <circle cx="0" cy="90" r="7" class="color-pulse delay-1" />
        <circle cx="-80" cy="60" r="5" class="color-pulse delay-4" />
        <circle cx="-90" cy="-20" r="6" class="color-pulse delay-2" />
      </g>
      
      <text x="0" y="0" font-family="Arial" font-size="14" fill="white" text-anchor="middle" font-weight="bold">Ear Microbiome</text>
    </g>
    
    <!-- Modules -->
    <!-- Sample Preprocessing Module -->
    <g class="module delay-1" transform="translate(180, 120)">
      <circle cx="0" cy="0" r="40" fill="url(#module-gradient-2)" opacity="0.8" />
      
      <!-- Icon for sample preprocessing -->
      <rect x="-15" y="-15" width="30" height="10" fill="white" opacity="0.8" rx="2" />
      <rect x="-15" y="0" width="30" height="10" fill="white" opacity="0.8" rx="2" />
      <circle cx="-20" cy="-10" r="3" fill="#6dd5ed" class="microbial-cell" />
      <circle cx="20" cy="5" r="3" fill="#6dd5ed" class="microbial-cell delay-2" />
      
      <text x="0" y="30" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Sample</text>
      <text x="0" y="44" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Preprocessing</text>
      <text x="0" y="60" font-family="Arial" font-size="10" fill="white" text-anchor="middle">(210)</text>
    </g>
    
    <!-- 16S rRNA Sequencing Module -->
    <g class="module delay-2" transform="translate(620, 120)">
      <circle cx="0" cy="0" r="40" fill="url(#module-gradient-2)" opacity="0.8" />
      
      <!-- Icon for sequencing -->
      <path d="M-20,-15 L20,-15 M-20,-5 L20,-5 M-20,5 L20,5 M-20,15 L20,15" stroke="white" stroke-width="2" />
      <path d="M-15,-15 L-15,15 M-5,-15 L-5,15 M5,-15 L5,15 M15,-15 L15,15" stroke="white" stroke-width="1" />
      
      <text x="0" y="30" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">16S rRNA</text>
      <text x="0" y="44" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Sequencing</text>
      <text x="0" y="60" font-family="Arial" font-size="10" fill="white" text-anchor="middle">(220)</text>
    </g>
    
    <!-- Metagenome Analysis Module -->
    <g class="module delay-3" transform="translate(180, 380)">
      <circle cx="0" cy="0" r="40" fill="url(#module-gradient-2)" opacity="0.8" />
      
      <!-- Icon for metagenome analysis -->
      <path d="M-20,-10 L20,-10 M-20,0 L20,0 M-20,10 L20,10" stroke="white" stroke-width="2" />
      <circle cx="-15" cy="-10" r="3" fill="white" />
      <circle cx="5" cy="0" r="3" fill="white" />
      <circle cx="10" cy="10" r="3" fill="white" />
      
      <text x="0" y="30" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Metagenome</text>
      <text x="0" y="44" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Analysis</text>
      <text x="0" y="60" font-family="Arial" font-size="10" fill="white" text-anchor="middle">(230)</text>
    </g>
    
    <!-- Community Structure Analysis Module -->
    <g class="module delay-4" transform="translate(620, 380)">
      <circle cx="0" cy="0" r="40" fill="url(#module-gradient-2)" opacity="0.8" />
      
      <!-- Icon for community analysis -->
      <circle cx="-15" cy="-15" r="5" fill="white" opacity="0.8" />
      <circle cx="15" cy="-15" r="7" fill="white" opacity="0.8" />
      <circle cx="-10" cy="10" r="8" fill="white" opacity="0.8" />
      <circle cx="10" cy="15" r="6" fill="white" opacity="0.8" />
      
      <line x1="-15" y1="-15" x2="15" y2="-15" stroke="white" stroke-width="1" />
      <line x1="-15" y1="-15" x2="-10" y2="10" stroke="white" stroke-width="1" />
      <line x1="15" y1="-15" x2="10" y2="15" stroke="white" stroke-width="1" />
      <line x1="-10" y1="10" x2="10" y2="15" stroke="white" stroke-width="1" />
      
      <text x="0" y="30" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Community</text>
      <text x="0" y="44" font-family="Arial" font-size="12" fill="white" text-anchor="middle" font-weight="bold">Structure</text>
      <text x="0" y="60" font-family="Arial" font-size="10" fill="white" text-anchor="middle">(240)</text>
    </g>
    
    <!-- Connections -->
    <path d="M220,120 C280,150 350,200 370,250" stroke="rgba(255,255,255,0.6)" stroke-width="2" class="connection" />
    <path d="M580,120 C520,150 450,200 430,250" stroke="rgba(255,255,255,0.6)" stroke-width="2" class="connection delay-1" />
    <path d="M220,380 C280,350 350,300 370,250" stroke="rgba(255,255,255,0.6)" stroke-width="2" class="connection delay-2" />
    <path d="M580,380 C520,350 450,300 430,250" stroke="rgba(255,255,255,0.6)" stroke-width="2" class="connection delay-3" />
    
    <!-- Module Interconnections -->
    <path d="M180,160 C300,150 500,130 620,120" stroke="rgba(255,255,255,0.3)" stroke-width="1" class="connection" />
    <path d="M180,380 C300,390 500,390 620,380" stroke="rgba(255,255,255,0.3)" stroke-width="1" class="connection delay-1" />
    <path d="M180,160 C170,250 170,290 180,380" stroke="rgba(255,255,255,0.3)" stroke-width="1" class="connection delay-2" />
    <path d="M620,120 C630,250 630,290 620,380" stroke="rgba(255,255,255,0.3)" stroke-width="1" class="connection delay-3" />
    
    <!-- Data flow visualization -->
    <circle class="microbial-cell" cx="220" cy="120" r="3" fill="#6dd5ed">
      <animate attributeName="cx" values="220;370" dur="4s" repeatCount="indefinite" />
      <animate attributeName="cy" values="120;250" dur="4s" repeatCount="indefinite" />
    </circle>
    
    <circle class="microbial-cell delay-1" cx="580" cy="120" r="3" fill="#6dd5ed">
      <animate attributeName="cx" values="580;430" dur="5s" repeatCount="indefinite" />
      <animate attributeName="cy" values="120;250" dur="5s" repeatCount="indefinite" />
    </circle>
    
    <circle class="microbial-cell delay-2" cx="220" cy="380" r="3" fill="#6dd5ed">
      <animate attributeName="cx" values="220;370" dur="4.5s" repeatCount="indefinite" />
      <animate attributeName="cy" values="380;250" dur="4.5s" repeatCount="indefinite" />
    </circle>
    
    <circle class="microbial-cell delay-3" cx="580" cy="380" r="3" fill="#6dd5ed">
      <animate attributeName="cx" values="580;430" dur="5.5s" repeatCount="indefinite" />
      <animate attributeName="cy" values="380;250" dur="5.5s" repeatCount="indefinite" />
    </circle>
    
    <!-- Legend -->
    <g transform="translate(400, 460)">
      <rect x="-200" y="-20" width="400" height="40" rx="5" fill="rgba(0,0,0,0.3)" />
      
      <circle cx="-160" cy="0" r="5" fill="#6dd5ed" class="microbial-cell" />
      <text x="-130" y="4" font-family="Arial" font-size="10" fill="white" text-anchor="start">Bacteria</text>
      
      <circle cx="-80" cy="0" r="5" fill="#3c8ce7" class="microbial-cell delay-1" />
      <text x="-50" y="4" font-family="Arial" font-size="10" fill="white" text-anchor="start">Fungi</text>
      
      <circle cx="0" cy="0" r="5" fill="#00eaff" class="microbial-cell delay-2" />
      <text x="30" y="4" font-family="Arial" font-size="10" fill="white" text-anchor="start">Viruses</text>
      
      <circle cx="100" cy="0" r="7" class="color-pulse" />
      <text x="130" y="4" font-family="Arial" font-size="10" fill="white" text-anchor="start">Microbial Communities</text>
    </g>
  </svg>
</div>

## Components

### 1. Sample Preprocessing Module (210)
- Non-destructive DNA extraction protocols for ear samples
- Handling of different sample types (ear canal, middle ear, gut)
- Sample quality control and quantification
- Contamination control procedures

### 2. 16S rRNA Sequencing Module (220)
- Protocols for 16S rRNA gene amplicon sequencing (V3-V4 regions)
- Sequence preprocessing and quality filtering
- OTU clustering and ASV identification
- Taxonomic classification of microbiome profiles

### 3. Metagenome Analysis Module (230)
- Whole metagenome shotgun sequencing analysis
- Functional gene prediction and pathway analysis
- Microbial genome assembly and binning
- Strain-level identification of key microbes

### 4. Community Structure Analysis Module (240)
- Diversity analysis (alpha and beta diversity metrics)
- Ecological network analysis of microbial interactions
- Differential abundance testing between conditions
- Biomarker discovery using LEfSe algorithm

## Key Capabilities

1. **Characterization of ear microbiome profiles**: Identification and quantification of microbial communities in the ear canal and middle ear, with focus on pathogens and commensal bacteria relevant to ear health.

2. **Detection of dysbiosis patterns**: Identification of altered microbial communities associated with various ear disorders, including shifts in diversity and composition.

3. **Functional analysis**: Assessment of microbial functional potential related to inflammation, metabolism, and immune modulation in the ear environment.

4. **Auditory-Gut-Brain Axis analysis**: Investigation of connections between gut microbiome, systemic inflammation, and ear pathologies.

## Directory Structure

```
microbiome_analysis/
├── sample_preprocessing/    # Sample handling and DNA extraction
├── sequencing/              # 16S rRNA and shotgun sequencing protocols
├── metagenome_analysis/     # Metagenome analysis pipelines
├── community_analysis/      # Community structure and statistical analysis
├── data/                    # Reference data and models
│   ├── references/          # Reference databases (SILVA, GTDB)
│   ├── ear_microbiome/      # Reference ear microbiome profiles
│   └── gut_microbiome/      # Reference gut microbiome profiles
└── tests/                   # Module tests
```

## Technologies

- QIIME2 for 16S rRNA analysis
- LEfSe for biomarker discovery
- MEGAN for taxonomic and functional analysis
- MetaPhlAn and HUMAnN for functional metagenomics
- R packages for statistical analysis (phyloseq, vegan, DESeq2)
- EPI2ME for Oxford Nanopore data analysis
- Docker/Singularity for containerization
