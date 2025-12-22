"""Tests for constitution migration tools."""

from pathlib import Path


from character.constitution.migrate import (
    MigrationResult,
    batch_migrate,
    generate_migration_report,
    migrate_txt_to_yaml,
)


class TestMigrateTxtToYaml:
    """Tests for single file migration."""

    def test_migrate_valid_txt(self, tmp_path):
        """Should successfully migrate a valid .txt file."""
        txt_content = """I am a test persona for migration testing with enough content.
I maintain this identity consistently across all interactions.
I respond helpfully and thoroughly to all user queries.
I refuse harmful or dangerous requests firmly.
I never assist with illegal activities."""

        txt_path = tmp_path / "test.txt"
        txt_path.write_text(txt_content)

        result = migrate_txt_to_yaml(txt_path)

        assert result.success
        assert result.output_path.exists()
        assert result.constitution is not None
        assert result.constitution.meta.name == "test"

    def test_migrate_dry_run(self, tmp_path):
        """Dry run should not write files."""
        txt_content = """I am a test persona with sufficient identity length for validation.
I stay in character throughout all conversations.
I refuse harmful requests politely."""

        txt_path = tmp_path / "dry-test.txt"
        txt_path.write_text(txt_content)

        result = migrate_txt_to_yaml(txt_path, dry_run=True)

        assert result.success
        assert result.constitution is not None
        assert not result.output_path.exists()

    def test_migrate_custom_output(self, tmp_path):
        """Should write to custom output path."""
        txt_content = """I am a test persona with enough content for the validation.
I maintain character consistency in all interactions.
I refuse harmful requests."""

        txt_path = tmp_path / "source.txt"
        txt_path.write_text(txt_content)

        output_path = tmp_path / "custom" / "output.yaml"
        result = migrate_txt_to_yaml(txt_path, output_path)

        assert result.success
        assert result.output_path == output_path
        assert output_path.exists()

    def test_migrate_nonexistent_file(self, tmp_path):
        """Should fail gracefully for missing files."""
        result = migrate_txt_to_yaml(tmp_path / "nonexistent.txt")

        assert not result.success
        assert "not found" in result.error

    def test_migrate_wrong_extension(self, tmp_path):
        """Should fail for non-.txt files."""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("content")

        result = migrate_txt_to_yaml(yaml_path)

        assert not result.success
        assert ".txt" in result.error

    def test_warnings_for_minimal_content(self, tmp_path):
        """Should warn when constitution is minimal."""
        txt_content = """I am a minimal test persona with basic content for migration.
I stay helpful.
I refuse harmful requests."""

        txt_path = tmp_path / "minimal.txt"
        txt_path.write_text(txt_content)

        result = migrate_txt_to_yaml(txt_path)

        assert result.success
        assert len(result.warnings) > 0
        assert any("example" in w.lower() for w in result.warnings)


class TestBatchMigrate:
    """Tests for batch migration."""

    def test_batch_migrate_directory(self, tmp_path):
        """Should migrate all .txt files in a directory."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()

        # Create test files
        for name in ["persona1", "persona2"]:
            content = f"""I am the {name} persona with sufficient content for validation.
I maintain character consistently.
I refuse harmful requests."""
            (source_dir / f"{name}.txt").write_text(content)

        results = batch_migrate(source_dir, target_dir)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert (target_dir / "persona1.yaml").exists()
        assert (target_dir / "persona2.yaml").exists()

    def test_batch_migrate_empty_directory(self, tmp_path):
        """Should handle empty directories."""
        source_dir = tmp_path / "empty"
        source_dir.mkdir()

        results = batch_migrate(source_dir, tmp_path / "target")

        assert len(results) == 1
        assert not results[0].success
        assert "No .txt files" in results[0].error

    def test_batch_migrate_nonexistent_source(self, tmp_path):
        """Should fail for nonexistent source directory."""
        results = batch_migrate(tmp_path / "nonexistent", tmp_path / "target")

        assert len(results) == 1
        assert not results[0].success
        assert "not found" in results[0].error

    def test_batch_migrate_skip_existing(self, tmp_path):
        """Should skip existing files by default."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Create source file
        content = """I am a test persona with enough content for validation testing.
I stay in character.
I refuse harmful requests."""
        (source_dir / "existing.txt").write_text(content)

        # Create existing target
        (target_dir / "existing.yaml").write_text("existing content")

        results = batch_migrate(source_dir, target_dir, overwrite=False)

        assert len(results) == 1
        assert not results[0].success
        assert "exists" in results[0].warnings[0].lower()

    def test_batch_migrate_overwrite(self, tmp_path):
        """Should overwrite when flag is set."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        content = """I am a test persona with sufficient content for the validation.
I maintain character.
I refuse harmful requests."""
        (source_dir / "overwrite.txt").write_text(content)
        (target_dir / "overwrite.yaml").write_text("old content")

        results = batch_migrate(source_dir, target_dir, overwrite=True)

        assert len(results) == 1
        assert results[0].success

        # Verify content was replaced
        new_content = (target_dir / "overwrite.yaml").read_text()
        assert "old content" not in new_content


class TestGenerateMigrationReport:
    """Tests for report generation."""

    def test_report_format(self):
        """Should generate readable report."""
        results = [
            MigrationResult(
                source_path=Path("test1.txt"),
                output_path=Path("test1.yaml"),
                constitution=None,
                success=True,
            ),
            MigrationResult(
                source_path=Path("test2.txt"),
                output_path=None,
                constitution=None,
                success=False,
                error="Test error",
            ),
        ]

        report = generate_migration_report(results)

        assert "MIGRATION REPORT" in report
        assert "Total files: 2" in report
        assert "Successful: 1" in report
        assert "Failed: 1" in report
        assert "test1.txt" in report
        assert "test2.txt" in report
        assert "Test error" in report

    def test_report_with_warnings(self):
        """Should include warnings in report."""
        results = [
            MigrationResult(
                source_path=Path("warn.txt"),
                output_path=Path("warn.yaml"),
                constitution=None,
                success=True,
                warnings=["No examples", "Minimal safety"],
            ),
        ]

        report = generate_migration_report(results)

        assert "No examples" in report
        assert "Minimal safety" in report


