"""Unit tests for the sample MCP server functions."""
from __future__ import annotations

import pytest

from llming_models.tools.mcp.sample_server import (
    CATEGORIES,
    CUSTOMERS_DB,
    PRODUCTS_DB,
    call_tool,
    list_tools,
)


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


class TestListTools:
    """Tests for list_tools."""

    @pytest.mark.asyncio
    async def test_returns_five_tools(self):
        tools = await list_tools()
        assert len(tools) == 5

    @pytest.mark.asyncio
    async def test_tool_names(self):
        tools = await list_tools()
        names = {t.name for t in tools}
        assert names == {"search_products", "get_product_details", "list_categories", "get_customer", "check_inventory"}

    @pytest.mark.asyncio
    async def test_all_tools_have_schemas(self):
        tools = await list_tools()
        for t in tools:
            assert t.inputSchema is not None
            assert "type" in t.inputSchema


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------


class TestSearchProducts:
    """Tests for call_tool('search_products', ...)."""

    @pytest.mark.asyncio
    async def test_search_by_name(self):
        result = await call_tool("search_products", {"query": "headphones"})
        assert len(result) == 1
        text = result[0].text
        assert "Wireless Headphones" in text
        assert "P001" in text

    @pytest.mark.asyncio
    async def test_search_by_category(self):
        result = await call_tool("search_products", {"query": "electronics", "category": "Electronics"})
        text = result[0].text
        assert "Electronics" in text

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        result = await call_tool("search_products", {"query": "nonexistent_xyz"})
        assert len(result) == 1
        assert "No products found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_max_results(self):
        # Search "e" which matches many products
        result = await call_tool("search_products", {"query": "e", "max_results": 2})
        text = result[0].text
        # Count actual product lines (lines starting with "- P")
        product_lines = [line for line in text.split("\n") if line.strip().startswith("- P")]
        assert len(product_lines) <= 2

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self):
        result = await call_tool("search_products", {"query": "HEADPHONES"})
        text = result[0].text
        assert "Wireless Headphones" in text


# ---------------------------------------------------------------------------
# get_product_details
# ---------------------------------------------------------------------------


class TestGetProductDetails:
    """Tests for call_tool('get_product_details', ...)."""

    @pytest.mark.asyncio
    async def test_found_product(self):
        result = await call_tool("get_product_details", {"product_id": "P001"})
        text = result[0].text
        assert "Wireless Headphones" in text
        assert "$79.99" in text
        assert "In Stock" in text

    @pytest.mark.asyncio
    async def test_not_found(self):
        result = await call_tool("get_product_details", {"product_id": "P999"})
        text = result[0].text
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_details_contain_all_fields(self):
        result = await call_tool("get_product_details", {"product_id": "P003"})
        text = result[0].text
        assert "ID:" in text
        assert "Name:" in text
        assert "Category:" in text
        assert "Price:" in text
        assert "Stock:" in text
        assert "Status:" in text


# ---------------------------------------------------------------------------
# list_categories
# ---------------------------------------------------------------------------


class TestListCategories:
    """Tests for call_tool('list_categories')."""

    @pytest.mark.asyncio
    async def test_returns_all_categories(self):
        result = await call_tool("list_categories", {})
        text = result[0].text
        for cat in CATEGORIES:
            assert cat in text

    @pytest.mark.asyncio
    async def test_includes_product_counts(self):
        result = await call_tool("list_categories", {})
        text = result[0].text
        # Electronics has 3 products: P001, P002, P004
        assert "Electronics: 3" in text
        # Office has 3 products: P003, P005, P008
        assert "Office: 3" in text
        # Kitchen has 2 products: P006, P007
        assert "Kitchen: 2" in text


# ---------------------------------------------------------------------------
# get_customer
# ---------------------------------------------------------------------------


class TestGetCustomer:
    """Tests for call_tool('get_customer', ...)."""

    @pytest.mark.asyncio
    async def test_by_customer_id(self):
        result = await call_tool("get_customer", {"customer_id": "C001"})
        text = result[0].text
        assert "John Smith" in text
        assert "Gold" in text

    @pytest.mark.asyncio
    async def test_by_email(self):
        result = await call_tool("get_customer", {"email": "jane@example.com"})
        text = result[0].text
        assert "Jane Doe" in text
        assert "Silver" in text

    @pytest.mark.asyncio
    async def test_not_found(self):
        result = await call_tool("get_customer", {"customer_id": "C999"})
        text = result[0].text
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_not_found_bad_email(self):
        result = await call_tool("get_customer", {"email": "nobody@example.com"})
        text = result[0].text
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_empty_params_not_found(self):
        result = await call_tool("get_customer", {})
        text = result[0].text
        assert "not found" in text

    @pytest.mark.asyncio
    async def test_customer_id_takes_priority(self):
        # Both customer_id and email provided; customer_id should be used
        result = await call_tool("get_customer", {"customer_id": "C001", "email": "jane@example.com"})
        text = result[0].text
        assert "John Smith" in text  # C001 is John


# ---------------------------------------------------------------------------
# check_inventory
# ---------------------------------------------------------------------------


class TestCheckInventory:
    """Tests for call_tool('check_inventory', ...)."""

    @pytest.mark.asyncio
    async def test_all_inventory(self):
        result = await call_tool("check_inventory", {})
        text = result[0].text
        assert "Inventory Report" in text
        # Should contain all products
        for pid in PRODUCTS_DB:
            assert pid in text

    @pytest.mark.asyncio
    async def test_low_stock_only(self):
        result = await call_tool("check_inventory", {"low_stock_only": True})
        text = result[0].text
        # Products with stock < 100: P003 (75), P004 (30)
        assert "P004" in text  # 30 units
        assert "P003" in text  # 75 units
        # P001 (150) should NOT be in low stock
        assert "P001" not in text

    @pytest.mark.asyncio
    async def test_filter_by_category(self):
        result = await call_tool("check_inventory", {"category": "Kitchen"})
        text = result[0].text
        assert "Coffee Mug" in text
        assert "Water Bottle" in text
        # Non-kitchen items should not be present
        assert "Wireless Headphones" not in text

    @pytest.mark.asyncio
    async def test_low_stock_and_category(self):
        result = await call_tool("check_inventory", {"low_stock_only": True, "category": "Office"})
        text = result[0].text
        # Only Office products with stock < 100: P003 (75)
        assert "P003" in text
        assert "P005" not in text  # 200 units, not low stock

    @pytest.mark.asyncio
    async def test_no_match(self):
        # Low stock in Kitchen — all kitchen items have stock >= 100
        result = await call_tool("check_inventory", {"low_stock_only": True, "category": "Kitchen"})
        text = result[0].text
        assert "No products match" in text

    @pytest.mark.asyncio
    async def test_sorted_by_stock(self):
        result = await call_tool("check_inventory", {"category": "Electronics"})
        text = result[0].text
        lines = [l for l in text.split("\n") if l.strip().startswith("- P")]
        # Extract stock numbers to verify ascending order
        stocks = []
        for line in lines:
            # Format: "- P00X Name: NNN units ..."
            parts = line.split(":")
            if len(parts) >= 2:
                units_part = parts[1].strip().split(" ")[0]
                stocks.append(int(units_part))
        assert stocks == sorted(stocks)


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    """Tests for calling an unknown tool."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await call_tool("nonexistent_tool", {})
        text = result[0].text
        assert "Unknown tool" in text
