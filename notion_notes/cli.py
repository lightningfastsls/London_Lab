"""Click CLI entry point for Notion Atomic Notes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from notion_notes.claude_client import ClaudeClient
from notion_notes.commands.atomize import AtomizeConfig, Atomizer
from notion_notes.commands.link import LinkConfig, Linker
from notion_notes.commands.move import Mover
from notion_notes.commands.process import ProcessConfig, Processor
from notion_notes.commands.tag import TagConfig, Tagger
from notion_notes.config import load_config
from notion_notes.migration import MigrationConfig, NoteMigrator
from notion_notes.notion_client import NotionClientWrapper


@click.group()
@click.option("--dry-run", is_flag=True, help="Show what would happen without writing to Notion.")
@click.option(
    "--env-file",
    type=click.Path(exists=False),
    default=".env",
    help="Path to .env file (default: .env).",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, dry_run: bool, env_file: str, verbose: bool) -> None:
    """Notion Atomic Notes — process, atomize, link, and tag your knowledge base."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run
    ctx.obj["env_file"] = env_file


@cli.command("list-pages")
@click.argument("database_id")
@click.option("--limit", default=10, help="Maximum pages to list.")
@click.pass_context
def list_pages(ctx: click.Context, database_id: str, limit: int) -> None:
    """List pages in a Notion database (smoke test)."""
    env_path = Path(ctx.obj["env_file"])
    cfg = load_config(env_path=env_path if env_path.is_file() else None, dry_run=ctx.obj["dry_run"])
    client = NotionClientWrapper(
        token=cfg.notion_token,
        dry_run=cfg.dry_run,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )
    pages = client.query_database(database_id, page_size=limit)
    click.echo(f"Found {len(pages)} page(s):")
    for page in pages:
        click.echo(f"  - {page.title} ({page.id})")


def _parse_comma_list(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated string into a tuple, stripping whitespace."""
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@cli.command("migrate")
@click.option("--source", required=True, help="Notes database ID (source).")
@click.option("--target", required=True, help="Knowledge Base database ID (target).")
@click.option(
    "--require-course/--no-require-course",
    default=True,
    help="Only migrate pages with a UNI Course (default: yes).",
)
@click.option("--include-projects", default=None, help="Comma-separated course names to include.")
@click.option("--include-pages", default=None, help="Comma-separated page titles to include.")
@click.option("--exclude-pages", default=None, help="Comma-separated page titles to exclude.")
@click.pass_context
def migrate(
    ctx: click.Context,
    source: str,
    target: str,
    require_course: bool,
    include_projects: str | None,
    include_pages: str | None,
    exclude_pages: str | None,
) -> None:
    """Migrate pages from Notes DB to Knowledge Base DB."""
    env_path = Path(ctx.obj["env_file"])
    cfg = load_config(env_path=env_path if env_path.is_file() else None, dry_run=ctx.obj["dry_run"])
    client = NotionClientWrapper(
        token=cfg.notion_token,
        dry_run=cfg.dry_run,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )

    migration_cfg = MigrationConfig(
        source_db_id=source,
        target_db_id=target,
        dry_run=cfg.dry_run,
        require_course=require_course,
        include_projects=_parse_comma_list(include_projects),
        include_pages=_parse_comma_list(include_pages),
        exclude_pages=_parse_comma_list(exclude_pages),
    )

    migrator = NoteMigrator(client, migration_cfg)
    report = migrator.migrate()

    click.echo("\n" + report.summary())

    # Save JSON report
    report_path = Path("migration_report.json")
    report_data = {
        "total_source": report.total_source,
        "filtered_out": report.filtered_out,
        "migrated": report.migrated,
        "skipped_dry_run": report.skipped_dry_run,
        "errors": report.errors,
        "id_mapping": report.id_mapping,
        "relations_remapped": report.relations_remapped,
        "relations_skipped": report.relations_skipped,
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    click.echo(f"\nReport saved to {report_path}")


def _load_cfg_and_clients(
    ctx: click.Context,
) -> tuple["NotionNotesConfig", NotionClientWrapper, ClaudeClient]:
    """Shared helper: load config, build Notion and Claude clients."""
    from notion_notes.config import NotionNotesConfig  # for type hint only

    env_path = Path(ctx.obj["env_file"])
    cfg = load_config(
        env_path=env_path if env_path.is_file() else None,
        dry_run=ctx.obj["dry_run"],
    )
    notion = NotionClientWrapper(
        token=cfg.notion_token,
        dry_run=cfg.dry_run,
        rate_limit_rps=cfg.notion_rate_limit_rps,
    )
    claude = ClaudeClient(
        api_key=cfg.anthropic_api_key,
        default_model=cfg.claude_model,
    )
    return cfg, notion, claude


@cli.command("tag")
@click.option("--page", default=None, help="Tag a single page by ID.")
@click.option("--untagged", is_flag=True, help="Tag all pages with empty Tags.")
@click.option("--retag-all", is_flag=True, help="Re-tag all pages in the KB.")
@click.option(
    "--taxonomy",
    type=click.Path(),
    default="taxonomy.json",
    help="Path to taxonomy JSON (default: taxonomy.json).",
)
@click.pass_context
def tag(
    ctx: click.Context,
    page: str | None,
    untagged: bool,
    retag_all: bool,
    taxonomy: str,
) -> None:
    """Tag KB pages with domain and tags using Claude."""
    cfg, notion, claude = _load_cfg_and_clients(ctx)

    if not cfg.kb_database_id:
        raise click.ClickException(
            "KB_DATABASE_ID not set. Add it to your .env file or environment."
        )

    tag_config = TagConfig(taxonomy_path=Path(taxonomy))
    tagger = Tagger(
        notion=notion,
        claude=claude,
        config=tag_config,
        dry_run=cfg.dry_run,
    )

    # Determine which pages to tag
    if page:
        pages = [notion.get_page(page)]
    elif untagged:
        pages = notion.query_database(
            cfg.kb_database_id,
            filter={"property": "Tags", "multi_select": {"is_empty": True}},
        )
    elif retag_all:
        pages = notion.query_database(cfg.kb_database_id)
    else:
        raise click.ClickException(
            "Specify one of: --page PAGE_ID, --untagged, or --retag-all"
        )

    if not pages:
        click.echo("No pages to tag.")
        return

    click.echo(f"Tagging {len(pages)} page(s){'  [DRY RUN]' if cfg.dry_run else ''}...")
    results = tagger.tag_pages(pages)

    # Print results
    errors = 0
    for r in results:
        if r.error:
            click.echo(f"  ERROR  {r.title}: {r.error}")
            errors += 1
        else:
            new_str = f"  (new: {r.new_tags})" if r.new_tags else ""
            click.echo(f"  OK     {r.title} -> {r.domain} {r.tags}{new_str}")

    click.echo(f"\nDone: {len(results) - errors} tagged, {errors} errors.")


@cli.command("atomize")
@click.option("--page", default=None, help="Atomize a single page by ID.")
@click.option("--unprocessed", is_flag=True, help="Atomize all pages with Status = 'Raw'.")
@click.option("--recent", default=None, type=int, help="Atomize the N most recently created pages.")
@click.option(
    "--taxonomy",
    type=click.Path(),
    default="taxonomy.json",
    help="Path to taxonomy JSON (default: taxonomy.json).",
)
@click.pass_context
def atomize(
    ctx: click.Context,
    page: str | None,
    unprocessed: bool,
    recent: int | None,
    taxonomy: str,
) -> None:
    """Atomize raw class notes into atomic concept pages using Claude."""
    cfg, notion, claude = _load_cfg_and_clients(ctx)

    if not cfg.kb_database_id:
        raise click.ClickException(
            "KB_DATABASE_ID not set. Add it to your .env file or environment."
        )

    atomize_config = AtomizeConfig(taxonomy_path=Path(taxonomy))
    atomizer = Atomizer(
        notion=notion,
        claude=claude,
        config=atomize_config,
        dry_run=cfg.dry_run,
    )

    # Determine which pages to atomize
    if page:
        pages = [notion.get_page(page)]
    elif unprocessed:
        pages = notion.query_database(
            cfg.kb_database_id,
            filter={"property": "Status", "select": {"equals": "Raw"}},
        )
    elif recent is not None:
        pages = notion.query_database(
            cfg.kb_database_id,
            sorts=[{"timestamp": "created_time", "direction": "descending"}],
            page_size=recent,
        )
        # Skip already-atomized (Atomizer also checks, but filter here for cleaner output)
        pages = [
            p for p in pages
            if NotionClientWrapper.extract_property(p.properties, "Status") != "Atomized"
        ]
    else:
        raise click.ClickException(
            "Specify one of: --page PAGE_ID, --unprocessed, or --recent N"
        )

    if not pages:
        click.echo("No pages to atomize.")
        return

    click.echo(f"Atomizing {len(pages)} page(s){'  [DRY RUN]' if cfg.dry_run else ''}...")
    results = atomizer.atomize_pages(pages)

    # Print results
    errors = 0
    for r in results:
        if r.error:
            click.echo(f"  ERROR  {r.title}: {r.error}")
            errors += 1
        else:
            n = len(r.concepts_created)
            titles = [c["title"] for c in r.concepts_created]
            click.echo(f"  OK     {r.title} -> {n} concept(s): {', '.join(titles)}")

    click.echo(f"\nDone: {len(results) - errors} atomized, {errors} errors.")


@cli.command("link")
@click.option("--page", default=None, help="Link a single page by ID.")
@click.option("--all-unlinked", is_flag=True, help="Link all pages with Status != 'Linked'.")
@click.option("--recent", default=None, type=int, help="Link the N most recently created pages.")
@click.pass_context
def link(
    ctx: click.Context,
    page: str | None,
    all_unlinked: bool,
    recent: int | None,
) -> None:
    """Discover and create semantic connections between KB pages using Claude."""
    cfg, notion, claude = _load_cfg_and_clients(ctx)

    if not cfg.kb_database_id:
        raise click.ClickException(
            "KB_DATABASE_ID not set. Add it to your .env file or environment."
        )

    link_config = LinkConfig()
    linker = Linker(
        notion=notion,
        claude=claude,
        config=link_config,
        dry_run=cfg.dry_run,
    )

    # Always fetch all pages for candidate comparison
    all_pages = notion.query_database(cfg.kb_database_id)

    # Determine which pages to link
    if page:
        pages = [notion.get_page(page)]
    elif all_unlinked:
        pages = [
            p for p in all_pages
            if NotionClientWrapper.extract_property(p.properties, "Status") != "Linked"
        ]
    elif recent is not None:
        sorted_pages = sorted(
            all_pages, key=lambda p: p.created_time, reverse=True
        )
        pages = sorted_pages[:recent]
    else:
        raise click.ClickException(
            "Specify one of: --page PAGE_ID, --all-unlinked, or --recent N"
        )

    if not pages:
        click.echo("No pages to link.")
        return

    click.echo(
        f"Linking {len(pages)} page(s) against {len(all_pages)} KB pages"
        f"{'  [DRY RUN]' if cfg.dry_run else ''}..."
    )
    results = linker.link_pages(pages, all_pages)

    # Print results
    errors = 0
    total_connections = 0
    for r in results:
        if r.error:
            click.echo(f"  ERROR  {r.page_title}: {r.error}")
            errors += 1
        else:
            n = len(r.connections_found)
            total_connections += n
            if n > 0:
                conn_strs = [
                    f"{c.connection_type}" for c in r.connections_found
                ]
                click.echo(
                    f"  OK     {r.page_title} -> {n} connection(s): "
                    f"{', '.join(conn_strs)} "
                    f"({r.candidates_evaluated} evaluated, {r.skipped_cached} cached)"
                )
            else:
                click.echo(
                    f"  OK     {r.page_title} -> 0 connections "
                    f"({r.candidates_evaluated} evaluated, {r.skipped_cached} cached)"
                )

    click.echo(
        f"\nDone: {len(results) - errors} linked, "
        f"{total_connections} connections found, {errors} errors."
    )


@cli.command("process")
@click.option("--recent", default=None, type=int, help="Process the N most recently created pages.")
@click.option("--unprocessed", is_flag=True, help="Process all pages with Status = 'Raw'.")
@click.option(
    "--taxonomy",
    type=click.Path(),
    default="taxonomy.json",
    help="Path to taxonomy JSON (default: taxonomy.json).",
)
@click.pass_context
def process(
    ctx: click.Context,
    recent: int | None,
    unprocessed: bool,
    taxonomy: str,
) -> None:
    """Run full pipeline: tag -> atomize -> link on KB pages."""
    cfg, notion, claude = _load_cfg_and_clients(ctx)

    if not cfg.kb_database_id:
        raise click.ClickException(
            "KB_DATABASE_ID not set. Add it to your .env file or environment."
        )

    # Determine which pages to process
    if recent is not None:
        pages = notion.query_database(
            cfg.kb_database_id,
            sorts=[{"timestamp": "created_time", "direction": "descending"}],
            page_size=recent,
        )
    elif unprocessed:
        pages = notion.query_database(
            cfg.kb_database_id,
            filter={"property": "Status", "select": {"equals": "Raw"}},
        )
    else:
        raise click.ClickException(
            "Specify one of: --recent N or --unprocessed"
        )

    if not pages:
        click.echo("No pages to process.")
        return

    click.echo(
        f"Processing {len(pages)} page(s){'  [DRY RUN]' if cfg.dry_run else ''}..."
    )

    process_config = ProcessConfig(
        tag_config=TagConfig(taxonomy_path=Path(taxonomy)),
        atomize_config=AtomizeConfig(taxonomy_path=Path(taxonomy)),
        link_config=LinkConfig(),
        kb_database_id=cfg.kb_database_id,
    )
    processor = Processor(
        notion=notion,
        claude=claude,
        config=process_config,
        dry_run=cfg.dry_run,
    )
    report = processor.process(pages)
    click.echo("\n" + report.summary())


@cli.command("move")
@click.option("--page", required=True, help="Page ID to move from Notes inbox to KB.")
@click.pass_context
def move(ctx: click.Context, page: str) -> None:
    """Move a page from Notes inbox to Knowledge Base."""
    cfg, notion, _ = _load_cfg_and_clients(ctx)

    if not cfg.kb_database_id:
        raise click.ClickException(
            "KB_DATABASE_ID not set. Add it to your .env file or environment."
        )

    mover = Mover(
        notion=notion,
        target_db_id=cfg.kb_database_id,
        dry_run=cfg.dry_run,
    )
    result = mover.move_page(page)

    if result.error:
        click.echo(f"ERROR: {result.error}")
        raise SystemExit(1)
    else:
        click.echo(
            f"Moved '{result.source_title}' -> {result.new_page_id}"
            f"{'  [DRY RUN]' if cfg.dry_run else ''}"
        )
