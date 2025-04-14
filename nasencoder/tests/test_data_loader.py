import pytest
import numpy as np
import os
import tempfile
import time
from unittest.mock import patch, MagicMock
from nasencoder.data.loader import GenomicDataLoader, NascentDataset


class TestGenomicDataLoader:
    """Tests for the GenomicDataLoader class."""
    
    @pytest.fixture
    def mock_pysam(self):
        """Mock pysam module."""
        with patch('nasencoder.data.loader.pysam') as mock_pysam:
            # Mock FastaFile
            mock_fasta = MagicMock()
            mock_fasta.references = ['chr1', 'chr2']
            mock_fasta.get_reference_length.return_value = 10000
            mock_fasta.fetch.return_value = 'ACGTACGTACGTACGT' * 16  # 256 bp sequence
            mock_pysam.FastaFile.return_value = mock_fasta
            
            # Mock AlignmentFile
            mock_bam = MagicMock()
            mock_read = MagicMock()
            mock_read.is_secondary = False
            mock_read.is_supplementary = False
            mock_read.is_reverse = False
            mock_read.get_reference_positions.return_value = list(range(100, 200))
            mock_bam.fetch.return_value = [mock_read]
            mock_pysam.AlignmentFile.return_value = mock_bam
            
            yield mock_pysam
    
    @pytest.fixture
    def test_files(self):
        """Create temporary test files."""
        with tempfile.NamedTemporaryFile(suffix='.fa') as fasta_file, \
             tempfile.NamedTemporaryFile(suffix='.bam') as bam_file:
            yield fasta_file.name, bam_file.name
    
    def test_init(self, mock_pysam, test_files):
        """Test initialization."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        assert loader.fasta_file == fasta_file
        assert loader.bam_file == bam_file
        assert loader.region_size == 256
        assert loader.chromosomes == ['chr1', 'chr2']
    
    def test_one_hot_encode_sequence(self, mock_pysam, test_files):
        """Test sequence one-hot encoding."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        sequence = 'ACGT'
        encoding = loader.one_hot_encode_sequence(sequence)
        
        assert encoding.shape == (4, 4)
        assert np.array_equal(encoding[0], [1, 0, 0, 0])  # A
        assert np.array_equal(encoding[1], [0, 1, 0, 0])  # C
        assert np.array_equal(encoding[2], [0, 0, 1, 0])  # G
        assert np.array_equal(encoding[3], [0, 0, 0, 1])  # T
    
    def test_get_coverage(self, mock_pysam, test_files):
        """Test coverage calculation."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        pos_coverage, neg_coverage = loader.get_coverage('chr1', 100, 200)
        
        assert pos_coverage.shape == (100,)
        assert neg_coverage.shape == (100,)
        assert np.sum(pos_coverage) + np.sum(neg_coverage) == pytest.approx(1.0)
    
    def test_get_region(self, mock_pysam, test_files):
        """Test region extraction."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        region = loader.get_region('chr1', 500)
        
        assert region['chrom'] == 'chr1'
        assert region['center'] == 500
        assert 'start' in region
        assert 'end' in region
        assert region['sequence'].shape == (256, 4)
        assert region['pos_coverage'].shape == (256,)
        assert region['neg_coverage'].shape == (256,)
    
    def test_get_batch_regions(self, mock_pysam, test_files):
        """Test batch region extraction."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        regions = [('chr1', 500), ('chr2', 600)]
        batch_regions = loader.get_batch_regions(regions)
        
        assert len(batch_regions) == 2
        assert batch_regions[0]['chrom'] == 'chr1'
        assert batch_regions[0]['center'] == 500
        assert batch_regions[1]['chrom'] == 'chr2'
        assert batch_regions[1]['center'] == 600
    
    def test_benchmark_loading(self, mock_pysam, test_files):
        """Test benchmarking function."""
        fasta_file, bam_file = test_files
        loader = GenomicDataLoader(fasta_file, bam_file)
        
        result = loader.benchmark_loading(n_regions=10)
        
        assert 'n_regions' in result
        assert 'single_thread_time' in result
        assert 'batch_time' in result
        assert 'regions_per_second_single' in result
        assert 'regions_per_second_batch' in result
        assert 'speedup' in result


class TestNascentDataset:
    """Tests for the NascentDataset class."""
    
    @pytest.fixture
    def mock_genomic_loader(self):
        """Mock GenomicDataLoader."""
        with patch('nasencoder.data.loader.GenomicDataLoader') as MockLoader:
            mock_loader = MagicMock()
            mock_loader.chromosomes = ['chr1', 'chr2']
            mock_loader.fasta.get_reference_length.return_value = 10000
            
            # Mock get_region
            def mock_get_region(chrom, center):
                return {
                    'chrom': chrom,
                    'center': center,
                    'start': center - 128,
                    'end': center + 128,
                    'sequence': np.random.rand(256, 4),
                    'pos_coverage': np.random.rand(256),
                    'neg_coverage': np.random.rand(256),
                }
            
            mock_loader.get_region.side_effect = mock_get_region
            
            # Mock get_batch_regions
            def mock_get_batch_regions(regions):
                return [mock_get_region(r[0], r[1]) for r in regions]
            
            mock_loader.get_batch_regions.side_effect = mock_get_batch_regions
            
            MockLoader.return_value = mock_loader
            yield MockLoader
    
    @pytest.fixture
    def cell_type_data(self):
        """Test cell type data."""
        with tempfile.NamedTemporaryFile(suffix='.fa') as fasta_file, \
             tempfile.NamedTemporaryFile(suffix='.bam') as bam_file1, \
             tempfile.NamedTemporaryFile(suffix='.bam') as bam_file2:
            yield {
                'cell_type1': (fasta_file.name, bam_file1.name),
                'cell_type2': (fasta_file.name, bam_file2.name),
            }
    
    def test_init(self, mock_genomic_loader, cell_type_data):
        """Test initialization."""
        dataset = NascentDataset(cell_type_data)
        
        assert dataset.cell_type_data == cell_type_data
        assert dataset.region_size == 256
        assert dataset.batch_size == 32
        assert dataset.mask_ratio == (0.15, 0.4)
        assert set(dataset.cell_types) == {'cell_type1', 'cell_type2'}
        assert set(dataset.loaders.keys()) == {'cell_type1', 'cell_type2'}
        assert dataset.cell_type_encoding['cell_type1'].shape == (2,)
        assert dataset.cell_type_encoding['cell_type2'].shape == (2,)
    
    def test_random_mask(self, mock_genomic_loader, cell_type_data):
        """Test random masking."""
        dataset = NascentDataset(cell_type_data)
        
        # Test 1D data
        data_1d = np.random.rand(256)
        masked_1d, mask_indices_1d = dataset.random_mask(data_1d, mask_ratio=0.2)
        
        assert masked_1d.shape == data_1d.shape
        assert len(mask_indices_1d) == int(256 * 0.2)
        assert np.all(masked_1d[mask_indices_1d] == 0)
        
        # Test 2D data
        data_2d = np.random.rand(256, 4)
        masked_2d, mask_indices_2d = dataset.random_mask(data_2d, mask_ratio=0.3)
        
        assert masked_2d.shape == data_2d.shape
        assert len(mask_indices_2d) == int(256 * 0.3)
        assert np.all(masked_2d[mask_indices_2d] == 0)
    
    def test_get_batch(self, mock_genomic_loader, cell_type_data):
        """Test batch creation."""
        dataset = NascentDataset(cell_type_data)
        
        regions_by_cell_type = {
            'cell_type1': [('chr1', 500), ('chr1', 600)],
            'cell_type2': [('chr2', 500), ('chr2', 600)],
        }
        
        batch = dataset.get_batch(regions_by_cell_type)
        
        assert 'sequence' in batch
        assert 'masked_sequence' in batch
        assert 'pos_coverage' in batch
        assert 'masked_pos_coverage' in batch
        assert 'neg_coverage' in batch
        assert 'masked_neg_coverage' in batch
        assert 'cell_type' in batch
        assert 'mask_indices' in batch
        assert 'metadata' in batch
        
        assert len(batch['sequence']) == 4
        assert len(batch['masked_sequence']) == 4
        assert len(batch['pos_coverage']) == 4
        assert len(batch['masked_pos_coverage']) == 4
        assert len(batch['neg_coverage']) == 4
        assert len(batch['masked_neg_coverage']) == 4
        assert len(batch['cell_type']) == 4
        assert len(batch['mask_indices']) == 4
        assert len(batch['metadata']) == 4
    
    def test_get_random_batch(self, mock_genomic_loader, cell_type_data):
        """Test random batch creation."""
        dataset = NascentDataset(cell_type_data, batch_size=4)
        
        batch = dataset.get_random_batch()
        
        assert 'sequence' in batch
        assert 'masked_sequence' in batch
        assert 'pos_coverage' in batch
        assert 'masked_pos_coverage' in batch
        assert 'neg_coverage' in batch
        assert 'masked_neg_coverage' in batch
        assert 'cell_type' in batch
        assert 'mask_indices' in batch
        assert 'metadata' in batch
        
        assert len(batch['sequence']) == 4
        assert len(batch['masked_sequence']) == 4
        assert len(batch['pos_coverage']) == 4
        assert len(batch['masked_pos_coverage']) == 4
        assert len(batch['neg_coverage']) == 4
        assert len(batch['masked_neg_coverage']) == 4
        assert len(batch['cell_type']) == 4
        assert len(batch['mask_indices']) == 4
        assert len(batch['metadata']) == 4
        
        # Test that sequence is properly masked
        for i in range(4):
            mask_indices = batch['mask_indices'][i]
            masked_sequence = batch['masked_sequence'][i]
            assert np.all(masked_sequence[mask_indices] == 0)