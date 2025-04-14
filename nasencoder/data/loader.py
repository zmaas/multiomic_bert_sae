import numpy as np
import pysam
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor
import time


class GenomicDataLoader:
    """
    Load genomic data from FASTA and BAM/CRAM files.
    """
    def __init__(self, fasta_file, bam_file, region_size=256):
        """
        Initialize the data loader.
        
        Args:
            fasta_file (str): Path to the FASTA file containing reference genome
            bam_file (str): Path to the BAM/CRAM file containing read alignments
            region_size (int): Size of regions to extract (default: 256bp)
        """
        self.fasta_file = fasta_file
        self.bam_file = bam_file
        self.region_size = region_size
        
        # Open the files
        self.fasta = pysam.FastaFile(fasta_file)
        self.bam = pysam.AlignmentFile(bam_file, "rb")
        
        # Get the list of chromosomes/contigs
        self.chromosomes = self.fasta.references
    
    def __del__(self):
        """Close file handles on destruction"""
        try:
            if hasattr(self, 'fasta'):
                self.fasta.close()
            if hasattr(self, 'bam'):
                self.bam.close()
        except Exception:
            pass
    
    def one_hot_encode_sequence(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence (str): DNA sequence string
            
        Returns:
            np.ndarray: One-hot encoded sequence of shape (len(sequence), 4)
        """
        # Map nucleotides to indices
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Initialize the encoding matrix
        encoding = np.zeros((len(sequence), 5), dtype=np.float32)
        
        # Fill in the encoding
        for i, nucleotide in enumerate(sequence.upper()):
            if nucleotide in mapping:
                encoding[i, mapping[nucleotide]] = 1.0
            else:
                # Handle ambiguous bases by setting them to N
                encoding[i, 4] = 1.0
                
        return encoding[:, :4]  # Drop the N channel for simplicity
    
    def get_coverage(self, chrom, start, end):
        """
        Get read coverage for a genomic region.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position (0-based)
            end (int): End position (0-based)
            
        Returns:
            tuple: (positive_strand_coverage, negative_strand_coverage)
        """
        # Initialize coverage arrays
        pos_coverage = np.zeros(end - start, dtype=np.float32)
        neg_coverage = np.zeros(end - start, dtype=np.float32)
        
        # Extract reads in the region
        for read in self.bam.fetch(chrom, start, end):
            # Skip secondary and supplementary alignments
            if read.is_secondary or read.is_supplementary:
                continue
            
            # Get read positions that overlap with our region
            read_positions = read.get_reference_positions()
            
            # Filter positions within our region
            positions_in_region = [p - start for p in read_positions if start <= p < end]
            
            # Add coverage
            if positions_in_region:
                if read.is_reverse:
                    for pos in positions_in_region:
                        neg_coverage[pos] += 1
                else:
                    for pos in positions_in_region:
                        pos_coverage[pos] += 1
        
        # Normalize coverage (sum to 1 as mentioned in the spec)
        total_coverage = np.sum(pos_coverage) + np.sum(neg_coverage)
        if total_coverage > 0:
            pos_coverage /= total_coverage
            neg_coverage /= total_coverage
            
        return pos_coverage, neg_coverage
    
    def get_region(self, chrom, center, pad_with_n=True):
        """
        Extract a fixed-size region centered at a given position.
        
        Args:
            chrom (str): Chromosome name
            center (int): Center position (0-based)
            pad_with_n (bool): Whether to pad with N's if region extends beyond chromosome
            
        Returns:
            dict: Region data containing sequence, coverage, and metadata
        """
        # Calculate region boundaries
        half_size = self.region_size // 2
        start = max(0, center - half_size)
        end = start + self.region_size
        
        # Check if region extends beyond chromosome length
        chrom_length = self.fasta.get_reference_length(chrom)
        if end > chrom_length:
            if pad_with_n:
                # Adjust start and end, sequence will be padded later
                end = chrom_length
                start = max(0, end - self.region_size)
            else:
                # Return None if region extends beyond chromosome
                return None
        
        # Extract sequence
        sequence = self.fasta.fetch(chrom, start, end)
        
        # Pad with N's if needed
        if len(sequence) < self.region_size and pad_with_n:
            pad_left = max(0, center - half_size - start)
            pad_right = self.region_size - len(sequence) - pad_left
            sequence = 'N' * pad_left + sequence + 'N' * pad_right
        
        # One-hot encode the sequence
        encoded_sequence = self.one_hot_encode_sequence(sequence)
        
        # Get coverage
        pos_coverage, neg_coverage = self.get_coverage(chrom, start, end)
        
        # Handle padding for coverage if needed
        if len(pos_coverage) < self.region_size and pad_with_n:
            pad_left = max(0, center - half_size - start)
            pad_right = self.region_size - len(pos_coverage) - pad_left
            
            pos_coverage = np.pad(pos_coverage, (pad_left, pad_right), 'constant')
            neg_coverage = np.pad(neg_coverage, (pad_left, pad_right), 'constant')
        
        # Return the region data
        return {
            'chrom': chrom,
            'center': center,
            'start': start,
            'end': end,
            'sequence': encoded_sequence,
            'pos_coverage': pos_coverage,
            'neg_coverage': neg_coverage,
        }
    
    def get_batch_regions(self, regions):
        """
        Get multiple regions in parallel.
        
        Args:
            regions (list): List of (chrom, center) tuples
            
        Returns:
            list: List of region data dictionaries
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda r: self.get_region(r[0], r[1]), 
                regions
            ))
        
        # Filter out None results
        return [r for r in results if r is not None]
    
    def benchmark_loading(self, n_regions=1000, random_seed=42):
        """
        Benchmark data loading speed.
        
        Args:
            n_regions (int): Number of regions to load
            random_seed (int): Random seed for reproducibility
            
        Returns:
            dict: Benchmark results
        """
        np.random.seed(random_seed)
        
        # Generate random regions
        regions = []
        for _ in range(n_regions):
            chrom = np.random.choice(self.chromosomes)
            chrom_length = self.fasta.get_reference_length(chrom)
            center = np.random.randint(self.region_size // 2, chrom_length - self.region_size // 2)
            regions.append((chrom, center))
        
        # Benchmark single-threaded loading
        start_time = time.time()
        for chrom, center in regions:
            self.get_region(chrom, center)
        single_time = time.time() - start_time
        
        # Benchmark multi-threaded loading
        start_time = time.time()
        self.get_batch_regions(regions)
        batch_time = time.time() - start_time
        
        return {
            'n_regions': n_regions,
            'single_thread_time': single_time,
            'batch_time': batch_time,
            'regions_per_second_single': n_regions / single_time,
            'regions_per_second_batch': n_regions / batch_time,
            'speedup': single_time / batch_time
        }


class NascentDataset:
    """
    Dataset class for nascent transcription data.
    
    Handles multiple cell types and batching.
    """
    def __init__(self, cell_type_data, region_size=256, batch_size=32, mask_ratio=(0.15, 0.4)):
        """
        Initialize the dataset.
        
        Args:
            cell_type_data (dict): Dictionary mapping cell type names to (fasta_file, bam_file) tuples
            region_size (int): Size of regions to extract
            batch_size (int): Batch size
            mask_ratio (tuple): Range of masking ratio for random masking
        """
        self.cell_type_data = cell_type_data
        self.region_size = region_size
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        
        # Create a loader for each cell type
        self.loaders = {}
        self.cell_types = list(cell_type_data.keys())
        
        for cell_type, (fasta_file, bam_file) in cell_type_data.items():
            self.loaders[cell_type] = GenomicDataLoader(fasta_file, bam_file, region_size)
        
        # One-hot encode cell types
        self.cell_type_encoding = {
            cell_type: np.zeros(len(self.cell_types), dtype=np.float32)
            for cell_type in self.cell_types
        }
        
        for i, cell_type in enumerate(self.cell_types):
            self.cell_type_encoding[cell_type][i] = 1.0
    
    def random_mask(self, data, mask_ratio=None):
        """
        Apply random masking to the data.
        
        Args:
            data (np.ndarray): Data to mask
            mask_ratio (float): Ratio of positions to mask
            
        Returns:
            tuple: (masked_data, mask_indices)
        """
        if mask_ratio is None:
            # Random masking ratio in the specified range
            mask_ratio = np.random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        
        # Generate mask indices
        n_positions = data.shape[0]
        n_mask = int(n_positions * mask_ratio)
        
        # Mask positions in chunks
        chunk_size = max(1, n_positions // 10)  # Adjust chunk size as needed
        n_chunks = n_positions // chunk_size
        
        # Select random chunks to mask
        n_chunks_to_mask = max(1, int(n_mask / chunk_size))
        chunks_to_mask = np.random.choice(n_chunks, n_chunks_to_mask, replace=False)
        
        # Create mask indices
        mask_indices = []
        for chunk in chunks_to_mask:
            start = chunk * chunk_size
            end = min(start + chunk_size, n_positions)
            mask_indices.extend(list(range(start, end)))
        
        # Truncate to desired mask size
        mask_indices = mask_indices[:n_mask]
        
        # Create a copy of the data
        masked_data = data.copy()
        
        # Mask the data
        if len(data.shape) == 1:
            # For 1D data (e.g., coverage)
            masked_data[mask_indices] = 0
        else:
            # For 2D data (e.g., sequence one-hot encoding)
            masked_data[mask_indices, :] = 0
        
        return masked_data, mask_indices
    
    def get_batch(self, regions_by_cell_type):
        """
        Get a batch of data.
        
        Args:
            regions_by_cell_type (dict): Dict mapping cell type to list of (chrom, center) tuples
            
        Returns:
            dict: Batch data
        """
        batch_data = {
            'sequence': [],
            'masked_sequence': [],
            'pos_coverage': [],
            'masked_pos_coverage': [],
            'neg_coverage': [],
            'masked_neg_coverage': [],
            'cell_type': [],
            'mask_indices': [],
            'metadata': []
        }
        
        for cell_type, regions in regions_by_cell_type.items():
            # Get loader for this cell type
            loader = self.loaders[cell_type]
            
            # Get regions data
            cell_regions = loader.get_batch_regions(regions)
            
            for region in cell_regions:
                # Get data
                sequence = region['sequence']
                pos_coverage = region['pos_coverage']
                neg_coverage = region['neg_coverage']
                
                # Apply masking
                masked_sequence, mask_indices_seq = self.random_mask(sequence)
                masked_pos_coverage, _ = self.random_mask(pos_coverage, mask_ratio=0.3)
                masked_neg_coverage, _ = self.random_mask(neg_coverage, mask_ratio=0.3)
                
                # Add to batch
                batch_data['sequence'].append(sequence)
                batch_data['masked_sequence'].append(masked_sequence)
                batch_data['pos_coverage'].append(pos_coverage)
                batch_data['masked_pos_coverage'].append(masked_pos_coverage)
                batch_data['neg_coverage'].append(neg_coverage)
                batch_data['masked_neg_coverage'].append(masked_neg_coverage)
                batch_data['cell_type'].append(self.cell_type_encoding[cell_type])
                batch_data['mask_indices'].append(mask_indices_seq)
                batch_data['metadata'].append({
                    'chrom': region['chrom'],
                    'center': region['center'],
                    'start': region['start'],
                    'end': region['end'],
                    'cell_type': cell_type
                })
        
        # Convert lists to numpy arrays
        for key in batch_data:
            if key != 'metadata' and key != 'mask_indices':
                batch_data[key] = np.array(batch_data[key])
        
        return batch_data
    
    def get_random_batch(self):
        """
        Get a random batch.
        
        Returns:
            dict: Batch data
        """
        regions_by_cell_type = {}
        regions_per_cell_type = self.batch_size // len(self.cell_types)
        
        for cell_type in self.cell_types:
            loader = self.loaders[cell_type]
            
            # Generate random regions
            regions = []
            for _ in range(regions_per_cell_type):
                chrom = np.random.choice(loader.chromosomes)
                chrom_length = loader.fasta.get_reference_length(chrom)
                center = np.random.randint(
                    loader.region_size // 2, 
                    chrom_length - loader.region_size // 2
                )
                regions.append((chrom, center))
            
            regions_by_cell_type[cell_type] = regions
        
        return self.get_batch(regions_by_cell_type)